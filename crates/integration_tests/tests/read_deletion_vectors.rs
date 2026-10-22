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

//! Integration tests for deletion vectors written by Spark.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::arrow::{ArrowReaderBuilder, schema_to_arrow_schema};
use iceberg::delete_vector::DeleteVector;
use iceberg::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
use iceberg::spec::{DataFile, DataFileFormat};
use iceberg::table::Table;
use iceberg::writer::base_writer::deletion_vector_writer::DeletionVectorWriterBuilder;
use iceberg::writer::base_writer::position_delete_writer::position_delete_schema;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, Runtime, TableIdent};
use iceberg_catalog_rest::RestCatalogBuilder;
use iceberg_integration_tests::get_test_fixture;
use iceberg_storage_opendal::OpenDalStorageFactory;

async fn load_table() -> Table {
    let catalog = RestCatalogBuilder::default()
        .with_storage_factory(Arc::new(OpenDalStorageFactory::S3 {
            customized_credential_load: None,
        }))
        .load("rest", get_test_fixture().catalog_config.clone())
        .await
        .unwrap();
    catalog
        .load_table(&TableIdent::from_strs(["default", "test_deletion_vectors"]).unwrap())
        .await
        .unwrap()
}

async fn plan(table: &Table) -> Vec<FileScanTask> {
    table
        .scan()
        .build()
        .unwrap()
        .plan_files()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap()
}

/// Decodes the deletion vector Spark wrote for `task`'s data file.
async fn spark_vector(table: &Table, task: &FileScanTask) -> DeleteVector {
    let [dv] = task.deletes() else {
        panic!("expected one deletion vector, got {:?}", task.deletes());
    };
    assert_eq!(dv.file_format(), DataFileFormat::Puffin);
    let puffin = table.file_io().new_input(dv.file_path()).unwrap();
    let bytes = puffin.read().await.unwrap();
    let start = dv.content_offset().unwrap() as usize;
    let end = start + dv.content_size_in_bytes().unwrap() as usize;
    let vector = DeleteVector::deserialize(&bytes[start..end]).unwrap();
    assert_eq!(Some(vector.len()), dv.record_count());
    vector
}

/// Reads `task`'s data file with `dv` as its only delete file.
async fn read_with(table: &Table, task: &FileScanTask, dv: &DataFile) -> Vec<RecordBatch> {
    let delete = FileScanTaskDeleteFile::builder()
        .with_file_path(dv.file_path().to_string())
        .with_file_size_in_bytes(dv.file_size_in_bytes())
        .with_file_type(dv.content_type())
        .with_file_format(dv.file_format())
        .with_partition_spec_id(table.metadata().default_partition_spec_id())
        .with_referenced_data_file(dv.referenced_data_file())
        .with_content_offset(dv.content_offset().map(|o| o as u64))
        .with_content_size_in_bytes(dv.content_size_in_bytes().map(|s| s as u64))
        .with_record_count(Some(dv.record_count()))
        .build()
        .unwrap();
    let task = FileScanTask::builder()
        .with_file_size_in_bytes(task.file_size_in_bytes())
        .with_start(task.start())
        .with_length(task.length())
        .with_record_count(task.record_count())
        .with_data_file_path(task.data_file_path().to_string())
        .with_data_file_format(task.data_file_format())
        .with_schema(task.schema_ref())
        .with_project_field_ids(task.project_field_ids().to_vec())
        .with_deletes(vec![delete])
        .with_case_sensitive(task.case_sensitive())
        .build()
        .unwrap();
    let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
    ArrowReaderBuilder::new(table.file_io().clone(), Runtime::current())
        .build()
        .read(tasks)
        .unwrap()
        .stream()
        .try_collect()
        .await
        .unwrap()
}

#[tokio::test]
async fn test_spark_deletion_vectors_reserialize() {
    let table = load_table().await;
    let tasks = plan(&table).await;
    assert_eq!(tasks.len(), 4);
    for task in &tasks {
        let vector = spark_vector(&table, task).await;
        let reserialized = DeleteVector::deserialize(&vector.serialize().unwrap()).unwrap();
        assert_eq!(
            reserialized.iter().collect::<Vec<_>>(),
            vector.iter().collect::<Vec<_>>()
        );
    }
}

/// Merges new positions into each of Spark's deletion vectors, as a writer must when it
/// replaces one, and reads the data file back with the merged vector alone.
#[tokio::test]
async fn test_merged_deletion_vector_replaces_spark_vector() {
    let table = load_table().await;
    let builder = DeletionVectorWriterBuilder::new(
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(
            format!("test-{}", uuid::Uuid::now_v7()),
            None,
            DataFileFormat::Puffin,
        ),
    );
    let schema = Arc::new(schema_to_arrow_schema(&position_delete_schema()).unwrap());
    for task in plan(&table).await {
        let existing = spark_vector(&table, &task).await;
        let existing_len = existing.len();
        let record_count = task.record_count().unwrap();
        let path = task.data_file_path();
        let new: Vec<i64> = (0..record_count as i64).filter(|p| p % 10 == 5).collect();

        let mut writer = builder.build(None).await.unwrap();
        writer.merge_existing(path, existing).unwrap();
        writer
            .write(
                RecordBatch::try_new(schema.clone(), vec![
                    Arc::new(StringArray::from(vec![path; new.len()])),
                    Arc::new(Int64Array::from(new.clone())),
                ])
                .unwrap(),
            )
            .await
            .unwrap();
        let [dv] = writer.close().await.unwrap().try_into().unwrap();

        // Each file holds 250 consecutive ids starting at a multiple of 10, so position p holds
        // an id ending in the same digit, and Spark deleted only ids ending in 0 and 1.
        assert_eq!(dv.record_count(), existing_len + new.len() as u64);
        let rows: usize = read_with(&table, &task, &dv)
            .await
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows as u64, record_count - dv.record_count());
    }
}
