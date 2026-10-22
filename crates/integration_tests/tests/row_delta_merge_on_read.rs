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

//! Integration tests for RowDelta committing delete files to tables Spark wrote.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::delete_vector::DeleteVector;
use iceberg::scan::FileScanTask;
use iceberg::spec::{DataFile, DataFileFormat};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::deletion_vector_writer::DeletionVectorWriterBuilder;
use iceberg::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, position_delete_schema,
};
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, TableIdent};
use iceberg_catalog_rest::{RestCatalog, RestCatalogBuilder};
use iceberg_integration_tests::get_test_fixture;
use iceberg_storage_opendal::OpenDalStorageFactory;
use parquet::file::properties::WriterProperties;

async fn catalog() -> RestCatalog {
    RestCatalogBuilder::default()
        .with_storage_factory(Arc::new(OpenDalStorageFactory::S3 {
            customized_credential_load: None,
        }))
        .load("rest", get_test_fixture().catalog_config.clone())
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

async fn row_count(table: &Table) -> usize {
    let batches: Vec<RecordBatch> = table
        .scan()
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    batches.iter().map(|b| b.num_rows()).sum()
}

/// The `(file_path, pos)` batch deleting the positions ending in 5 of every task's file.
fn deletes_ending_in_5(tasks: &[FileScanTask]) -> (RecordBatch, usize) {
    let mut paths = Vec::new();
    let mut positions = Vec::new();
    for task in tasks {
        for pos in (0..task.record_count().unwrap() as i64).filter(|p| p % 10 == 5) {
            paths.push(task.data_file_path().to_string());
            positions.push(pos);
        }
    }
    let schema = Arc::new(schema_to_arrow_schema(&position_delete_schema()).unwrap());
    let n = positions.len();
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from(paths)),
        Arc::new(Int64Array::from(positions)),
    ])
    .unwrap();
    (batch, n)
}

fn names(prefix: &str) -> DefaultFileNameGenerator {
    DefaultFileNameGenerator::new(
        format!("{prefix}-{}", uuid::Uuid::now_v7()),
        None,
        DataFileFormat::Puffin,
    )
}

/// Replaces Spark's deletion vectors with vectors that also delete new positions, in one
/// RowDelta commit, then reads the table back through the catalog.
#[tokio::test]
async fn test_row_delta_replaces_spark_deletion_vectors() {
    let catalog = catalog().await;
    let ident = TableIdent::from_strs(["default", "test_row_delta_v3"]).unwrap();
    let table = catalog.load_table(&ident).await.unwrap();
    let before = row_count(&table).await;
    let tasks = plan(&table).await;
    assert_eq!(tasks.len(), 4);

    let builder = DeletionVectorWriterBuilder::new(
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        names("row-delta"),
    );
    let mut writer = builder.build(None).await.unwrap();
    let mut replaced: Vec<DataFile> = Vec::new();
    for task in &tasks {
        let [old] = task.deletes() else {
            panic!("expected one deletion vector per data file");
        };
        let bytes = table
            .file_io()
            .new_input(old.file_path())
            .unwrap()
            .read()
            .await
            .unwrap();
        let start = old.content_offset().unwrap() as usize;
        let end = start + old.content_size_in_bytes().unwrap() as usize;
        writer
            .merge_existing(
                task.data_file_path(),
                DeleteVector::deserialize(&bytes[start..end]).unwrap(),
            )
            .unwrap();
        replaced.push(old_dv_as_data_file(&table, old.file_path(), start as i64).await);
    }
    let (batch, new_deletes) = deletes_ending_in_5(&tasks);
    writer.write(batch).await.unwrap();
    let added = writer.close().await.unwrap();
    assert_eq!(added.len(), 4);

    let tx = Transaction::new(&table);
    let tx = tx
        .row_delta()
        .add_delete_files(added)
        .remove_delete_files(replaced)
        .apply(tx)
        .unwrap();
    tx.commit(&catalog).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    // Spark deleted ids ending in 0; each file holds 250 consecutive ids starting at a
    // multiple of 10, so the positions ending in 5 are all still live.
    assert_eq!(row_count(&table).await, before - new_deletes);
    assert!(plan(&table).await.iter().all(|t| t.deletes().len() == 1));
}

/// Commits a position delete file to a v2 table Spark wrote.
#[tokio::test]
async fn test_row_delta_adds_position_deletes_to_v2_table() {
    let catalog = catalog().await;
    let ident = TableIdent::from_strs(["default", "test_row_delta_v2"]).unwrap();
    let table = catalog.load_table(&ident).await.unwrap();
    let before = row_count(&table).await;
    let tasks = plan(&table).await;

    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(WriterProperties::default(), position_delete_schema()),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new(
            format!("pos-{}", uuid::Uuid::now_v7()),
            None,
            DataFileFormat::Parquet,
        ),
    );
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling)
        .build(None)
        .await
        .unwrap();
    let mut sorted = tasks.clone();
    sorted.sort_by(|a, b| a.data_file_path().cmp(b.data_file_path()));
    let (batch, new_deletes) = deletes_ending_in_5(&sorted);
    writer.write(batch).await.unwrap();
    let files = writer.close().await.unwrap();

    let tx = Transaction::new(&table);
    let tx = tx.row_delta().add_delete_files(files).apply(tx).unwrap();
    tx.commit(&catalog).await.unwrap();

    let table = catalog.load_table(&ident).await.unwrap();
    assert_eq!(row_count(&table).await, before - new_deletes);
}

/// The table's manifest entry for the deletion vector at `path` and `offset`.
async fn old_dv_as_data_file(table: &Table, path: &str, offset: i64) -> DataFile {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let list = table.manifest_list_reader(snapshot).load().await.unwrap();
    for manifest_file in list.entries() {
        let manifest = table.manifest_reader().read(manifest_file).await.unwrap();
        for entry in manifest.entries() {
            let file = entry.data_file();
            if entry.is_alive() && file.file_path() == path && file.content_offset() == Some(offset)
            {
                return file.clone();
            }
        }
    }
    panic!("no live deletion vector {path} at {offset}");
}
