// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! This module provides `DeletionVectorWriter`, which writes v3 deletion vectors.
//!
//! The writer takes the same `(file_path, pos)` batches as
//! [`PositionDeleteFileWriter`](super::position_delete_writer::PositionDeleteFileWriter)
//! (see [`position_delete_schema`](super::position_delete_writer::position_delete_schema)).
//! On close it writes one Puffin file holding one `deletion-vector-v1` blob per referenced
//! data file, and returns one [`DataFile`] per blob.
//!
//! A v3 table allows at most one deletion vector per data file in a snapshot, so a new vector
//! must also hold the positions of the data file's existing one; see
//! [`DeletionVectorWriter::merge_existing`]. Replacing the existing vector in the commit is
//! the caller's job.

use std::collections::{BTreeMap, HashMap};

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;

use super::position_delete_writer::validate_position_delete_batch;
use crate::delete_vector::DeleteVector;
use crate::error::invalid_data;
use crate::io::FileIO;
use crate::metadata_columns::RESERVED_FIELD_ID_POS;
use crate::puffin::{
    Blob, CREATED_BY_PROPERTY, CompressionCodec, DELETION_VECTOR_V1, PuffinWriter,
};
use crate::spec::{DataContentType, DataFile, DataFileBuilder, DataFileFormat, PartitionKey};
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Blob property naming the data file a deletion vector applies to.
const REFERENCED_DATA_FILE_PROPERTY: &str = "referenced-data-file";
/// Blob property holding the number of deleted positions.
const CARDINALITY_PROPERTY: &str = "cardinality";

/// Builder for [`DeletionVectorWriter`].
#[derive(Clone, Debug)]
pub struct DeletionVectorWriterBuilder<L: LocationGenerator, F: FileNameGenerator> {
    file_io: FileIO,
    location_generator: L,
    file_name_generator: F,
}

impl<L, F> DeletionVectorWriterBuilder<L, F>
where
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Creates a builder whose writers name their Puffin file with `file_name_generator` and
    /// place it with `location_generator`.
    pub fn new(file_io: FileIO, location_generator: L, file_name_generator: F) -> Self {
        Self {
            file_io,
            location_generator,
            file_name_generator,
        }
    }
}

#[async_trait::async_trait]
impl<L, F> IcebergWriterBuilder for DeletionVectorWriterBuilder<L, F>
where
    L: LocationGenerator,
    F: FileNameGenerator,
{
    type R = DeletionVectorWriter<L, F>;

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        Ok(DeletionVectorWriter {
            builder: self.clone(),
            partition_key,
            vectors: Some(BTreeMap::new()),
        })
    }
}

/// Writes the deletion vectors of data files within one spec/partition.
#[derive(Debug)]
pub struct DeletionVectorWriter<L: LocationGenerator, F: FileNameGenerator> {
    builder: DeletionVectorWriterBuilder<L, F>,
    partition_key: Option<PartitionKey>,
    vectors: Option<BTreeMap<String, DeleteVector>>,
}

impl<L, F> DeletionVectorWriter<L, F>
where
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Adds the positions `existing` already deletes from `data_file_path`, so the vector
    /// written for that file replaces `existing`.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed.
    pub fn merge_existing(
        &mut self,
        data_file_path: impl Into<String>,
        existing: DeleteVector,
    ) -> Result<()> {
        let vectors = self.open_vectors()?;
        *vectors.entry(data_file_path.into()).or_default() |= existing;
        Ok(())
    }

    fn open_vectors(&mut self) -> Result<&mut BTreeMap<String, DeleteVector>> {
        self.vectors.as_mut().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Deletion vector writer is already closed.",
            )
        })
    }
}

#[async_trait::async_trait]
impl<L, F> IcebergWriter for DeletionVectorWriter<L, F>
where
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Adds a batch of `(file_path, pos)` records, in any order. Repeated records are kept once.
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        let vectors = self.open_vectors()?;
        validate_position_delete_batch(&batch)?;
        let paths = batch.column(0).as_string::<i32>();
        let positions = batch.column(1).as_primitive::<Int64Type>();
        for (path, pos) in paths.iter().zip(positions.values()) {
            // Both columns are validated as non-nullable.
            let path = path.expect("file_path is non-nullable");
            let pos = u64::try_from(*pos)
                .map_err(|_| invalid_data!("Position delete `pos` must not be negative: {pos}"))?;
            match vectors.get_mut(path) {
                Some(vector) => vector.insert(pos),
                None => vectors.entry(path.to_string()).or_default().insert(pos),
            };
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<Vec<DataFile>> {
        let vectors = self.vectors.take().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Deletion vector writer is already closed.",
            )
        })?;
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let builder = &self.builder;
        let location = builder.location_generator.generate_location(
            self.partition_key.as_ref(),
            &builder.file_name_generator.generate_file_name(),
        );
        let output = builder.file_io.new_output(&location)?;
        let properties = HashMap::from([(
            CREATED_BY_PROPERTY.to_string(),
            format!("Apache Iceberg Rust {}", env!("CARGO_PKG_VERSION")),
        )]);
        let mut puffin = PuffinWriter::new(&output, properties, false).await?;
        for (data_file_path, vector) in &vectors {
            let blob = Blob::builder()
                .r#type(DELETION_VECTOR_V1.to_string())
                .fields(vec![RESERVED_FIELD_ID_POS])
                // Not known until commit; the spec requires -1.
                .snapshot_id(-1)
                .sequence_number(-1)
                .data(vector.serialize()?)
                .properties(HashMap::from([
                    (
                        REFERENCED_DATA_FILE_PROPERTY.to_string(),
                        data_file_path.clone(),
                    ),
                    (CARDINALITY_PROPERTY.to_string(), vector.len().to_string()),
                ]))
                .build();
            // Deletion vector blobs must not be compressed.
            puffin.add(blob, CompressionCodec::None).await?;
        }
        let blobs = puffin.written_blobs_metadata().to_vec();
        let file_size_in_bytes = puffin.close().await?.size;

        vectors
            .into_iter()
            .zip(blobs)
            .map(|((data_file_path, vector), blob)| {
                let mut data_file = DataFileBuilder::default();
                data_file
                    .content(DataContentType::PositionDeletes)
                    .file_path(location.clone())
                    .file_format(DataFileFormat::Puffin)
                    .record_count(vector.len())
                    .file_size_in_bytes(file_size_in_bytes)
                    .referenced_data_file(Some(data_file_path))
                    .content_offset(Some(i64::try_from(blob.offset())?))
                    .content_size_in_bytes(Some(i64::try_from(blob.length())?));
                if let Some(pk) = self.partition_key.as_ref() {
                    data_file
                        .partition(pk.data().clone())
                        .partition_spec_id(pk.spec().spec_id());
                }
                data_file
                    .build()
                    .map_err(|e| invalid_data!("Failed to build deletion vector file: {e}"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::{Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use roaring::RoaringTreemap;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::ArrowReaderBuilder;
    use crate::puffin::PuffinReader;
    use crate::runtime::Runtime;
    use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
    use crate::spec::{
        Literal, NestedField, PartitionSpec, PrimitiveType, Schema, Struct, Transform, Type,
    };
    use crate::writer::base_writer::position_delete_writer::position_delete_schema;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };

    type TestBuilder =
        DeletionVectorWriterBuilder<DefaultLocationGenerator, DefaultFileNameGenerator>;

    fn builder(temp_dir: &TempDir) -> TestBuilder {
        DeletionVectorWriterBuilder::new(
            FileIO::new_with_fs(),
            DefaultLocationGenerator::with_data_location(
                temp_dir.path().to_str().unwrap().to_string(),
            ),
            DefaultFileNameGenerator::new("dv".to_string(), None, DataFileFormat::Puffin),
        )
    }

    fn delete_batch(rows: &[(&str, i64)]) -> RecordBatch {
        let schema =
            Arc::new(crate::arrow::schema_to_arrow_schema(&position_delete_schema()).unwrap());
        RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.0))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1))),
        ])
        .unwrap()
    }

    fn positions(vector: &DeleteVector) -> Vec<u64> {
        vector.iter().collect()
    }

    fn vector_of(positions: impl IntoIterator<Item = u64>) -> DeleteVector {
        DeleteVector::new(positions.into_iter().collect::<RoaringTreemap>())
    }

    /// Reads every blob of `path` back as `(blob metadata, decoded positions)`.
    async fn read_back(path: &str) -> (HashMap<String, String>, Vec<(BlobMeta, Vec<u64>)>) {
        let reader = PuffinReader::new(FileIO::new_with_fs().new_input(path).unwrap())
            .await
            .unwrap();
        let metadata = reader.file_metadata().await.unwrap().clone();
        let mut blobs = Vec::new();
        for blob_metadata in metadata.blobs() {
            let blob = reader.blob(blob_metadata).await.unwrap();
            let decoded = DeleteVector::deserialize(blob.data()).unwrap();
            blobs.push((
                BlobMeta {
                    blob_type: blob_metadata.blob_type().to_string(),
                    fields: blob_metadata.fields().to_vec(),
                    snapshot_id: blob_metadata.snapshot_id(),
                    sequence_number: blob_metadata.sequence_number(),
                    offset: blob_metadata.offset(),
                    length: blob_metadata.length(),
                    compression_codec: blob_metadata.compression_codec(),
                    properties: blob_metadata.properties().clone(),
                },
                positions(&decoded),
            ));
        }
        (metadata.properties().clone(), blobs)
    }

    #[derive(Debug, PartialEq)]
    struct BlobMeta {
        blob_type: String,
        fields: Vec<i32>,
        snapshot_id: i64,
        sequence_number: i64,
        offset: u64,
        length: u64,
        compression_codec: CompressionCodec,
        properties: HashMap<String, String>,
    }

    fn dv_blob_meta(data_file: &DataFile, referenced: &str, cardinality: u64) -> BlobMeta {
        BlobMeta {
            blob_type: DELETION_VECTOR_V1.to_string(),
            fields: vec![RESERVED_FIELD_ID_POS],
            snapshot_id: -1,
            sequence_number: -1,
            offset: data_file.content_offset().unwrap() as u64,
            length: data_file.content_size_in_bytes().unwrap() as u64,
            compression_codec: CompressionCodec::None,
            properties: HashMap::from([
                (
                    REFERENCED_DATA_FILE_PROPERTY.to_string(),
                    referenced.to_string(),
                ),
                (CARDINALITY_PROPERTY.to_string(), cardinality.to_string()),
            ]),
        }
    }

    #[tokio::test]
    async fn test_writes_one_blob_per_referenced_data_file() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = builder(&temp_dir).build(None).await?;
        writer
            .write(delete_batch(&[
                ("s3://b/data/f1.parquet", 7),
                ("s3://b/data/f0.parquet", 3),
                ("s3://b/data/f0.parquet", 1),
                ("s3://b/data/f0.parquet", 3),
            ]))
            .await?;
        writer
            .write(delete_batch(&[("s3://b/data/f0.parquet", 1 << 33)]))
            .await?;
        let data_files = writer.close().await?;

        let location = format!("{}/dv-00000.puffin", temp_dir.path().to_str().unwrap());
        let file_size_in_bytes = std::fs::metadata(&location).unwrap().len();
        let (properties, blobs) = read_back(&location).await;
        assert_eq!(blobs.len(), 2);
        let expected_file = |referenced: &str, cardinality: u64, blob: &BlobMeta| {
            DataFileBuilder::default()
                .content(DataContentType::PositionDeletes)
                .file_path(location.clone())
                .file_format(DataFileFormat::Puffin)
                .record_count(cardinality)
                .file_size_in_bytes(file_size_in_bytes)
                .referenced_data_file(Some(referenced.to_string()))
                .content_offset(Some(blob.offset as i64))
                .content_size_in_bytes(Some(blob.length as i64))
                .build()
                .unwrap()
        };
        assert_eq!(data_files, vec![
            expected_file("s3://b/data/f0.parquet", 3, &blobs[0].0),
            expected_file("s3://b/data/f1.parquet", 1, &blobs[1].0),
        ]);
        assert_eq!(blobs, vec![
            (
                dv_blob_meta(&data_files[0], "s3://b/data/f0.parquet", 3),
                vec![1, 3, 1 << 33]
            ),
            (
                dv_blob_meta(&data_files[1], "s3://b/data/f1.parquet", 1),
                vec![7]
            ),
        ]);
        assert!(
            properties[CREATED_BY_PROPERTY].starts_with("Apache Iceberg Rust "),
            "{properties:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_existing_keeps_existing_positions() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = builder(&temp_dir).build(None).await?;
        writer.merge_existing("s3://b/data/f0.parquet", vector_of([0, 2]))?;
        writer
            .write(delete_batch(&[
                ("s3://b/data/f0.parquet", 2),
                ("s3://b/data/f0.parquet", 4),
            ]))
            .await?;
        let data_files = writer.close().await?;

        assert_eq!(data_files.len(), 1);
        assert_eq!(data_files[0].record_count(), 3);
        let (_, blobs) = read_back(data_files[0].file_path()).await;
        assert_eq!(blobs[0].1, vec![0, 2, 4]);
        Ok(())
    }

    #[tokio::test]
    async fn test_close_without_deletes_writes_no_file() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = builder(&temp_dir).build(None).await?;
        assert_eq!(writer.close().await?, vec![]);
        assert_eq!(std::fs::read_dir(temp_dir.path()).unwrap().count(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_writer_sets_partition() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(2, "p", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(1)
            .add_partition_field("p", "p", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        let partition = Struct::from_iter([Some(Literal::int(3))]);
        let partition_key = PartitionKey::new(spec, schema, partition.clone());
        let mut writer = builder(&temp_dir).build(Some(partition_key)).await?;
        writer
            .write(delete_batch(&[("s3://b/data/p=3/f0.parquet", 5)]))
            .await?;
        let data_files = writer.close().await?;

        assert_eq!(data_files.len(), 1);
        assert_eq!(data_files[0].partition(), &partition);
        assert_eq!(data_files[0].partition_spec_id, 1);
        assert_eq!(
            data_files[0].file_path(),
            format!("{}/p=3/dv-00000.puffin", temp_dir.path().to_str().unwrap())
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_rejects_negative_position() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = builder(&temp_dir).build(None).await?;
        let err = writer
            .write(delete_batch(&[("s3://b/data/f0.parquet", -1)]))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        Ok(())
    }

    #[tokio::test]
    async fn test_rejects_use_after_close() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = builder(&temp_dir).build(None).await?;
        writer.close().await?;
        let batch = delete_batch(&[("s3://b/data/f0.parquet", 0)]);
        assert_eq!(
            writer.write(batch).await.unwrap_err().kind(),
            ErrorKind::Unexpected
        );
        assert_eq!(
            writer
                .merge_existing("s3://b/data/f0.parquet", vector_of([0]))
                .unwrap_err()
                .kind(),
            ErrorKind::Unexpected
        );
        assert_eq!(
            writer.close().await.unwrap_err().kind(),
            ErrorKind::Unexpected
        );
        Ok(())
    }

    /// The written vector, located only by the returned `DataFile`, filters the deleted rows
    /// out of a scan.
    #[tokio::test]
    async fn test_written_vector_is_applied_on_read() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let table_location = temp_dir.path().to_str().unwrap().to_string();
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));
        let data_file_path = format!("{table_location}/data.parquet");
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
            Int32Array::from_iter_values(1..=5),
        )])
        .unwrap();
        let mut parquet =
            ArrowWriter::try_new(File::create(&data_file_path).unwrap(), arrow_schema, None)
                .unwrap();
        parquet.write(&batch).unwrap();
        parquet.close().unwrap();

        let mut writer = builder(&temp_dir).build(None).await?;
        writer
            .write(delete_batch(&[(&data_file_path, 3), (&data_file_path, 1)]))
            .await?;
        let dv = writer.close().await?.remove(0);

        let task = FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(&data_file_path).unwrap().len())
            .with_start(0)
            .with_length(0)
            .with_record_count(Some(5))
            .with_data_file_path(data_file_path.clone())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(table_schema)
            .with_project_field_ids(vec![1])
            .with_deletes(vec![
                FileScanTaskDeleteFile::builder()
                    .with_file_path(dv.file_path().to_string())
                    .with_file_size_in_bytes(dv.file_size_in_bytes())
                    .with_file_type(dv.content_type())
                    .with_file_format(dv.file_format())
                    .with_partition_spec_id(dv.partition_spec_id)
                    .with_referenced_data_file(dv.referenced_data_file())
                    .with_content_offset(dv.content_offset().map(|o| o as u64))
                    .with_content_size_in_bytes(dv.content_size_in_bytes().map(|s| s as u64))
                    .with_record_count(Some(dv.record_count()))
                    .build()
                    .unwrap(),
            ])
            .with_case_sensitive(false)
            .build()
            .unwrap();
        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        let batches: Vec<RecordBatch> =
            ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
                .build()
                .read(tasks)?
                .stream()
                .try_collect()
                .await?;

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(ids, vec![1, 3, 5]);
        Ok(())
    }
}
