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

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{Result, invalid_data};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestEntry,
    ManifestFile, ManifestStatus, Operation,
};
use crate::table::Table;
use crate::transaction::ActionCommit;
use crate::transaction::action::StatelessAction;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::{Error, ErrorKind};

/// Transaction action for row-level modifications (UPDATE, DELETE, MERGE INTO), both
/// copy-on-write (add and remove data files) and merge-on-read (add and remove delete files).
///
/// Corresponds to `org.apache.iceberg.RowDelta` in the Java implementation.
#[derive(Clone)]
pub struct RowDeltaAction {
    added_data_files: Vec<DataFile>,
    removed_data_files: Vec<DataFile>,
    added_delete_files: Vec<DataFile>,
    removed_delete_files: Vec<DataFile>,
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    starting_snapshot_id: Option<i64>,
}

impl RowDeltaAction {
    pub(crate) fn new() -> Self {
        Self {
            added_data_files: vec![],
            removed_data_files: vec![],
            added_delete_files: vec![],
            removed_delete_files: vec![],
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            starting_snapshot_id: None,
        }
    }

    /// Add new data files (INSERT rows or Copy-on-Write rewritten files).
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Mark existing data files as deleted (Copy-on-Write mode).
    ///
    /// Corresponds to `removeRows(DataFile)` in the Java implementation.
    pub fn remove_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.removed_data_files.extend(data_files);
        self
    }

    /// Add delete files (Merge-on-Read): deletion vectors on a v3 table, position delete files
    /// on a v2 table, or equality delete files on either.
    ///
    /// A data file may have only one deletion vector in a snapshot, so a deletion vector that
    /// replaces an existing one must be added together with
    /// [`remove_delete_files`](Self::remove_delete_files) of the old one.
    pub fn add_delete_files(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_delete_files.extend(delete_files);
        self
    }

    /// Mark existing delete files, or deletion vectors, as removed.
    ///
    /// A deletion vector is identified by its Puffin file path and its `content_offset`.
    pub fn remove_delete_files(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.removed_delete_files.extend(delete_files);
        self
    }

    /// Set the commit UUID used for manifest file naming.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Attach custom key/value metadata to the snapshot summary.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Reject the commit if the table has advanced past `snapshot_id` (optimistic concurrency).
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.starting_snapshot_id = Some(snapshot_id);
        self
    }

    fn validate_added_delete_files(&self, table: &Table) -> Result<()> {
        let metadata = table.metadata();
        let format_version = metadata.format_version();
        if format_version == FormatVersion::V1 && !self.added_delete_files.is_empty() {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Delete files are not supported by format version 1",
            ));
        }
        let mut dv_targets = HashSet::new();
        for file in &self.added_delete_files {
            if file.content_type() == DataContentType::Data {
                return Err(invalid_data!(
                    "Data file {} cannot be added as a delete file",
                    file.file_path()
                ));
            }
            if metadata
                .partition_spec_by_id(file.partition_spec_id)
                .is_none()
            {
                return Err(invalid_data!(
                    "Delete file {} has unknown partition spec id {}",
                    file.file_path(),
                    file.partition_spec_id
                ));
            }
            if file.content_type() != DataContentType::PositionDeletes {
                continue;
            }
            match (format_version, file.file_format()) {
                (FormatVersion::V3, DataFileFormat::Puffin) => {
                    let referenced = match (
                        file.referenced_data_file(),
                        file.content_offset(),
                        file.content_size_in_bytes(),
                    ) {
                        (Some(referenced), Some(_), Some(_)) => referenced,
                        _ => {
                            return Err(invalid_data!(
                                "Deletion vector {} needs referenced_data_file, content_offset and content_size_in_bytes",
                                file.file_path()
                            ));
                        }
                    };
                    if !dv_targets.insert(referenced.clone()) {
                        return Err(invalid_data!(
                            "Data file {referenced} has more than one added deletion vector"
                        ));
                    }
                }
                (FormatVersion::V3, _) => {
                    return Err(invalid_data!(
                        "Format version 3 takes deletion vectors, not position delete file {}",
                        file.file_path()
                    ));
                }
                (_, DataFileFormat::Puffin) => {
                    return Err(invalid_data!(
                        "Deletion vector {} needs format version 3, the table is {format_version}",
                        file.file_path()
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Identifies a delete file: deletion vectors share a Puffin file, so its offset is part of it.
fn delete_file_key(file: &DataFile) -> (&str, Option<i64>) {
    (file.file_path(), file.content_offset())
}

#[async_trait]
impl StatelessAction for RowDeltaAction {
    async fn commit(&self, table: &Table) -> Result<ActionCommit> {
        if let Some(expected_snapshot_id) = self.starting_snapshot_id
            && table.metadata().current_snapshot_id() != Some(expected_snapshot_id)
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot commit RowDelta based on stale snapshot. Expected: {}, Current: {:?}",
                    expected_snapshot_id,
                    table.metadata().current_snapshot_id()
                ),
            ));
        }
        self.validate_added_delete_files(table)?;

        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            self.removed_data_files.clone(),
        );

        // Validate newly added data files (partition value type-checks, etc.).
        // removed_data_files are not re-validated: they are existing table files that were
        // already validated when originally committed. This matches Java's MergingSnapshotProducer.
        snapshot_producer.validate_added_data_files()?;

        let operation = RowDeltaOperation {
            adds_data_files: !self.added_data_files.is_empty(),
            removed_data_files: self.removed_data_files.clone(),
            added_delete_files: self.added_delete_files.clone(),
            removed_delete_files: self.removed_delete_files.clone(),
        };

        snapshot_producer
            .commit(operation, DefaultManifestProcess)
            .await
    }
}

struct RowDeltaOperation {
    adds_data_files: bool,
    removed_data_files: Vec<DataFile>,
    added_delete_files: Vec<DataFile>,
    removed_delete_files: Vec<DataFile>,
}

impl RowDeltaOperation {
    /// Fails if a data file would keep two live deletion vectors: an existing one that this
    /// commit does not remove, and one it adds.
    fn validate_one_dv_per_data_file(
        &self,
        live_dvs: &HashMap<String, (String, Option<i64>)>,
    ) -> Result<()> {
        let removed: HashSet<_> = self
            .removed_delete_files
            .iter()
            .map(delete_file_key)
            .collect();
        for file in &self.added_delete_files {
            if file.file_format() != DataFileFormat::Puffin {
                continue;
            }
            let Some(referenced) = file.referenced_data_file() else {
                continue;
            };
            if let Some((path, offset)) = live_dvs.get(&referenced)
                && !removed.contains(&(path.as_str(), *offset))
            {
                return Err(invalid_data!(
                    "Data file {referenced} already has deletion vector {path} (offset {offset:?}); remove it in the same commit"
                ));
            }
        }
        Ok(())
    }

    async fn write_added_delete_manifests(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let metadata = snapshot_produce.table.metadata();
        let mut by_spec: HashMap<i32, Vec<&DataFile>> = HashMap::new();
        for file in &self.added_delete_files {
            by_spec
                .entry(file.partition_spec_id)
                .or_default()
                .push(file);
        }
        let mut manifests = Vec::with_capacity(by_spec.len());
        for (spec_id, files) in by_spec {
            let spec = metadata
                .partition_spec_by_id(spec_id)
                .ok_or_else(|| invalid_data!("Unknown partition spec id {spec_id}"))?;
            let mut writer = snapshot_produce.new_manifest_writer(
                ManifestContentType::Deletes,
                metadata.current_schema().clone(),
                spec.clone(),
            )?;
            for file in files {
                // The sequence numbers are inherited from the snapshot at commit.
                writer.add_entry(
                    ManifestEntry::builder()
                        .status(ManifestStatus::Added)
                        .data_file(file.clone())
                        .build(),
                )?;
            }
            manifests.push(writer.write_manifest_file().await?);
        }
        Ok(manifests)
    }
}

impl SnapshotProduceOperation for RowDeltaOperation {
    /// Operation type based on Java `BaseRowDelta.operation()`:
    /// - Only data files added → `Append`
    /// - Delete files added and no data files added → `Delete`
    /// - Anything else → `Overwrite`
    fn operation(&self) -> Operation {
        let adds_deletes = !self.added_delete_files.is_empty();
        if self.adds_data_files && !adds_deletes && self.removed_data_files.is_empty() {
            Operation::Append
        } else if adds_deletes && !self.adds_data_files {
            Operation::Delete
        } else {
            Operation::Overwrite
        }
    }

    /// Delete entries are handled inside `existing_manifest` by rewriting the manifest.
    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    /// Returns manifest files for the new snapshot.
    ///
    /// For each manifest in the previous snapshot:
    /// - If it contains a live data file or delete file being removed: rewrite it with DELETED
    ///   entries for those files and EXISTING entries for the other live entries, preserving
    ///   their sequence numbers. Entries an earlier snapshot already deleted are dropped.
    /// - Otherwise: carry it forward unchanged.
    ///
    /// New delete manifests hold the added delete files, one per partition spec.
    ///
    /// This matches Java's `ManifestFilterManager.filterManifestWithDeletedFiles` logic.
    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let mut result = Vec::new();
        if let Some(snapshot) = snapshot_produce.table.metadata().current_snapshot() {
            let manifest_list = snapshot_produce
                .table
                .manifest_list_reader(snapshot)
                .load()
                .await?;

            let deleted_paths: HashSet<&str> = self
                .removed_data_files
                .iter()
                .map(|f| f.file_path())
                .collect();
            let deleted_deletes: HashSet<(&str, Option<i64>)> = self
                .removed_delete_files
                .iter()
                .map(delete_file_key)
                .collect();
            let mut removed_deletes = 0usize;
            let mut live_dvs = HashMap::new();

            for manifest_file in manifest_list.entries() {
                if !manifest_file.has_added_files() && !manifest_file.has_existing_files() {
                    continue;
                }

                let manifest = snapshot_produce
                    .table
                    .manifest_reader()
                    .read(manifest_file)
                    .await?;

                let is_removed = |entry: &ManifestEntry| match manifest_file.content {
                    ManifestContentType::Data => {
                        deleted_paths.contains(entry.data_file().file_path())
                    }
                    ManifestContentType::Deletes => {
                        deleted_deletes.contains(&delete_file_key(entry.data_file()))
                    }
                };

                if manifest_file.content == ManifestContentType::Deletes {
                    for entry in manifest.entries().iter().filter(|e| e.is_alive()) {
                        let file = entry.data_file();
                        if file.file_format() == DataFileFormat::Puffin
                            && let Some(referenced) = file.referenced_data_file()
                        {
                            live_dvs.insert(
                                referenced,
                                (file.file_path().to_string(), file.content_offset()),
                            );
                        }
                    }
                }

                let needs_rewrite = manifest
                    .entries()
                    .iter()
                    .any(|e| e.is_alive() && is_removed(e));

                if !needs_rewrite {
                    result.push(manifest_file.clone());
                    continue;
                }

                // Rewrite: deleted files → DELETED (new snapshot_id, original seq nums preserved),
                // surviving files → EXISTING (all original fields preserved).
                let metadata = snapshot_produce.table.metadata();
                let spec = metadata
                    .partition_spec_by_id(manifest_file.partition_spec_id)
                    .ok_or_else(|| {
                        invalid_data!(
                            "Unknown partition spec id {}",
                            manifest_file.partition_spec_id
                        )
                    })?;
                let mut writer = snapshot_produce.new_manifest_writer(
                    manifest_file.content,
                    metadata.current_schema().clone(),
                    spec.clone(),
                )?;
                for entry in manifest.entries().iter().filter(|e| e.is_alive()) {
                    if is_removed(entry) {
                        if manifest_file.content == ManifestContentType::Deletes {
                            removed_deletes += 1;
                        }
                        writer.add_delete_entry((**entry).clone())?;
                    } else {
                        writer.add_existing_entry((**entry).clone())?;
                    }
                }
                result.push(writer.write_manifest_file().await?);
            }

            if removed_deletes != deleted_deletes.len() {
                return Err(invalid_data!(
                    "{} of the {} delete files to remove are not live in the current snapshot",
                    deleted_deletes.len() - removed_deletes,
                    deleted_deletes.len()
                ));
            }
            self.validate_one_dv_per_data_file(&live_dvs)?;
        } else if !self.removed_delete_files.is_empty() {
            return Err(invalid_data!(
                "Cannot remove delete files from a table without a snapshot"
            ));
        }

        result.extend(self.write_added_delete_manifests(snapshot_produce).await?);
        Ok(result)
    }

    fn truncate_full_table(&self) -> bool {
        false
    }

    fn added_delete_files(&self) -> &[DataFile] {
        &self.added_delete_files
    }

    fn removed_delete_files(&self) -> &[DataFile] {
        &self.removed_delete_files
    }
}

#[cfg(test)]
mod tests {
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, MAIN_BRANCH,
        ManifestStatus, SnapshotRef, Struct, TableMetadataBuilder,
    };
    use crate::table::Table;
    use crate::transaction::Transaction;
    use crate::transaction::action::StatelessAction;
    use crate::transaction::tests::make_v2_minimal_table;
    use crate::{TableIdent, TableUpdate};

    fn make_data_file(table: &Table, path: &str, size: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(size)
            .record_count(10)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .build()
            .unwrap()
    }

    /// Build a table that has `snapshot` as its current snapshot, backed by the same FileIO.
    async fn table_with_snapshot(base: &Table, snapshot: crate::spec::Snapshot) -> Table {
        let updated_metadata =
            TableMetadataBuilder::new_from_metadata(base.metadata_ref().as_ref().clone(), None)
                .set_branch_snapshot(snapshot, MAIN_BRANCH)
                .unwrap()
                .build()
                .unwrap()
                .metadata;

        Table::builder()
            .metadata(updated_metadata)
            .metadata_location("s3://bucket/test/location/metadata/v2.json".to_string())
            .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
            .file_io(base.file_io().clone())
            .runtime(crate::test_utils::test_runtime())
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_row_delta_add_only() {
        let table = make_v2_minimal_table();
        let data_file = make_data_file(&table, "test/1.parquet", 100);
        let action = Transaction::new(&table)
            .row_delta()
            .add_data_files(vec![data_file]);

        let mut commit = StatelessAction::commit(&action, &table).await.unwrap();
        let updates = commit.take_updates();

        if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            assert_eq!(snapshot.summary().operation, crate::spec::Operation::Append);
        } else {
            panic!("expected AddSnapshot");
        }
    }

    #[tokio::test]
    async fn test_row_delta_with_snapshot_properties() {
        let table = make_v2_minimal_table();
        let data_file = make_data_file(&table, "test/1.parquet", 100);
        let mut props = std::collections::HashMap::new();
        props.insert("key".to_string(), "value".to_string());
        let action = Transaction::new(&table)
            .row_delta()
            .set_snapshot_properties(props)
            .add_data_files(vec![data_file]);

        let mut commit = StatelessAction::commit(&action, &table).await.unwrap();
        let updates = commit.take_updates();

        if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            assert_eq!(
                snapshot.summary().additional_properties.get("key").unwrap(),
                "value"
            );
        } else {
            panic!("expected AddSnapshot");
        }
    }

    #[tokio::test]
    async fn test_row_delta_validate_from_snapshot() {
        let table = make_v2_minimal_table();
        let data_file = make_data_file(&table, "test/1.parquet", 100);
        let action = Transaction::new(&table)
            .row_delta()
            .validate_from_snapshot(99999)
            .add_data_files(vec![data_file]);

        let result = StatelessAction::commit(&action, &table).await;
        match result {
            Ok(_) => panic!("expected DataInvalid error for stale snapshot"),
            Err(e) => assert_eq!(e.kind(), crate::ErrorKind::DataInvalid),
        }
    }

    #[tokio::test]
    async fn test_row_delta_empty_action() {
        let table = make_v2_minimal_table();
        assert!(
            StatelessAction::commit(&Transaction::new(&table).row_delta(), &table)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_row_delta_incompatible_partition_value() {
        let table = make_v2_minimal_table();
        let bad_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/bad.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(10)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::string("wrong"))]))
            .build()
            .unwrap();
        let action = Transaction::new(&table)
            .row_delta()
            .add_data_files(vec![bad_file]);
        assert!(StatelessAction::commit(&action, &table).await.is_err());
    }

    #[tokio::test]
    async fn test_row_delta_rejects_data_file_as_delete_file() {
        let table = make_v2_minimal_table();
        let file = make_data_file(&table, "test/delete.parquet", 100);
        let action = Transaction::new(&table)
            .row_delta()
            .add_delete_files(vec![file]);
        let Err(err) = StatelessAction::commit(&action, &table).await else {
            panic!("expected DataInvalid");
        };
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
    }

    /// End-to-end CoW test: append two files, then remove one via RowDelta.
    ///
    /// Verifies:
    /// - The removed file appears as DELETED with correct sequence numbers.
    /// - The surviving file appears as EXISTING with correct sequence numbers.
    /// - The new file appears as ADDED.
    /// - The snapshot summary counts `deleted-data-files = 1`.
    #[tokio::test]
    async fn test_row_delta_cow_manifest_rewrite() {
        let base_table = make_v2_minimal_table();

        // --- S1: append file-A and file-B ---
        let file_a = make_data_file(&base_table, "test/a.parquet", 100);
        let file_b = make_data_file(&base_table, "test/b.parquet", 200);

        let action1 = Transaction::new(&base_table)
            .fast_append()
            .add_data_files(vec![file_a.clone(), file_b.clone()]);
        let mut commit1 =
            crate::transaction::TransactionAction::commit(&action1, &mut (), &base_table)
                .await
                .unwrap();
        let updates1 = commit1.take_updates();

        let snapshot_s1 =
            if let TableUpdate::AddSnapshot { snapshot } = updates1.into_iter().next().unwrap() {
                snapshot
            } else {
                panic!("expected AddSnapshot");
            };

        let table_s1 = table_with_snapshot(&base_table, snapshot_s1).await;

        // --- S2: remove file-A (CoW), add file-C ---
        let file_c = make_data_file(&table_s1, "test/c.parquet", 300);
        let action2 = Transaction::new(&table_s1)
            .row_delta()
            .remove_data_files(vec![file_a.clone()])
            .add_data_files(vec![file_c.clone()]);
        let mut commit2 = StatelessAction::commit(&action2, &table_s1).await.unwrap();
        let updates2 = commit2.take_updates();

        let snapshot_s2 = if let TableUpdate::AddSnapshot { ref snapshot } = updates2[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot");
        };

        assert_eq!(
            snapshot_s2.summary().operation,
            crate::spec::Operation::Overwrite
        );

        // Verify snapshot summary metrics
        let props = &snapshot_s2.summary().additional_properties;
        assert_eq!(
            props.get("deleted-data-files").map(String::as_str),
            Some("1"),
            "summary should count 1 deleted file"
        );

        // Scan all manifest entries in S2
        let manifest_list = table_s1
            .manifest_list_reader(&SnapshotRef::new(snapshot_s2.clone()))
            .load()
            .await
            .unwrap();

        let mut found_deleted_a = false;
        let mut found_existing_b = false;
        let mut found_added_c = false;

        for manifest_file in manifest_list.entries() {
            let manifest = table_s1
                .manifest_reader()
                .read(manifest_file)
                .await
                .unwrap();
            for entry in manifest.entries() {
                match entry.data_file().file_path() {
                    "test/a.parquet" => {
                        assert_eq!(
                            entry.status(),
                            ManifestStatus::Deleted,
                            "file-A must be DELETED"
                        );
                        assert!(
                            entry.sequence_number().is_some(),
                            "DELETED entry must have sequence number"
                        );
                        assert!(
                            entry.file_sequence_number.is_some(),
                            "DELETED entry must have file sequence number"
                        );
                        found_deleted_a = true;
                    }
                    "test/b.parquet" => {
                        assert_eq!(
                            entry.status(),
                            ManifestStatus::Existing,
                            "file-B must be EXISTING"
                        );
                        assert!(
                            entry.sequence_number().is_some(),
                            "EXISTING entry must have sequence number"
                        );
                        found_existing_b = true;
                    }
                    "test/c.parquet" => {
                        found_added_c = true;
                    }
                    other => panic!("unexpected file in S2 manifests: {other}"),
                }
            }
        }

        assert!(found_deleted_a, "file-A should have a DELETED entry in S2");
        assert!(
            found_existing_b,
            "file-B should have an EXISTING entry in S2"
        );
        assert!(found_added_c, "file-C should have an ADDED entry in S2");
    }
}

#[cfg(test)]
mod merge_on_read_tests {
    use std::collections::HashMap;

    use futures::TryStreamExt;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
        ManifestContentType, ManifestStatus, NestedField, Operation, PartitionSpec, PrimitiveType,
        Schema, Struct, Transform, Type,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, ErrorKind, TableCreation, TableIdent};

    async fn table_in_catalog(catalog: &impl Catalog, version: FormatVersion) -> Table {
        let ident =
            TableIdent::from_strs([format!("ns-{}", uuid::Uuid::new_v4()), "t".to_string()])
                .unwrap();
        catalog
            .create_namespace(ident.namespace(), HashMap::new())
            .await
            .unwrap();
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        let spec = PartitionSpec::builder(schema.clone())
            .add_partition_field("x", "x", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        let creation = TableCreation::builder()
            .schema(schema)
            .partition_spec(spec)
            .name(ident.name().to_string())
            .format_version(version)
            .build();
        catalog
            .create_table(ident.namespace(), creation)
            .await
            .unwrap()
    }

    fn data_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(10)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    fn dv(puffin: &str, offset: i64, referenced: &str, cardinality: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(puffin.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(500)
            .record_count(cardinality)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .referenced_data_file(Some(referenced.to_string()))
            .content_offset(Some(offset))
            .content_size_in_bytes(Some(40))
            .build()
            .unwrap()
    }

    fn position_delete_file(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(2)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    /// `(content, status, path, offset)` of every entry of the current snapshot, sorted.
    async fn entries(
        table: &Table,
    ) -> Vec<(ManifestContentType, ManifestStatus, String, Option<i64>)> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let list = table.manifest_list_reader(snapshot).load().await.unwrap();
        let mut out = Vec::new();
        for manifest_file in list.entries() {
            let manifest = table.manifest_reader().read(manifest_file).await.unwrap();
            for e in manifest.entries() {
                out.push((
                    manifest_file.content,
                    e.status(),
                    e.data_file().file_path().to_string(),
                    e.data_file().content_offset(),
                ));
            }
        }
        out.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        out
    }

    #[tokio::test]
    async fn test_adds_and_replaces_deletion_vectors() {
        let catalog = new_memory_catalog().await;
        let table = table_in_catalog(&catalog, FormatVersion::V3).await;
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("d/a.parquet")])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let first = dv("d/dv1.puffin", 4, "d/a.parquet", 2);
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![first.clone()])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.summary().operation, Operation::Delete);
        let props = &snapshot.summary().additional_properties;
        assert_eq!(
            props.get("added-delete-files").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            props.get("added-position-deletes").map(String::as_str),
            Some("2")
        );
        // Only data manifests take a first_row_id.
        assert_eq!(table.metadata().next_row_id(), 10);

        // The scan attaches the vector to its data file.
        let tasks: Vec<_> = table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].deletes().len(), 1);
        assert_eq!(tasks[0].deletes()[0].file_path(), "d/dv1.puffin");

        // A second vector for the same data file needs the first one removed.
        let second = dv("d/dv2.puffin", 4, "d/a.parquet", 3);
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![second.clone()])
            .apply(tx)
            .unwrap();
        assert_eq!(
            tx.commit(&catalog).await.unwrap_err().kind(),
            ErrorKind::DataInvalid
        );

        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![second])
            .remove_delete_files(vec![first])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let props = &table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties;
        assert_eq!(
            props.get("removed-delete-files").map(String::as_str),
            Some("1")
        );
        assert_eq!(entries(&table).await, vec![
            (
                ManifestContentType::Data,
                ManifestStatus::Added,
                "d/a.parquet".to_string(),
                None
            ),
            (
                ManifestContentType::Deletes,
                ManifestStatus::Added,
                "d/dv2.puffin".to_string(),
                Some(4)
            ),
            (
                ManifestContentType::Deletes,
                ManifestStatus::Deleted,
                "d/dv1.puffin".to_string(),
                Some(4)
            ),
        ]);
        let tasks: Vec<_> = table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(tasks[0].deletes().len(), 1);
        assert_eq!(tasks[0].deletes()[0].file_path(), "d/dv2.puffin");
    }

    #[tokio::test]
    async fn test_vectors_sharing_a_puffin_file_are_removed_by_offset() {
        let catalog = new_memory_catalog().await;
        let table = table_in_catalog(&catalog, FormatVersion::V3).await;
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("d/a.parquet"), data_file("d/b.parquet")])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let dv_a = dv("d/dv.puffin", 4, "d/a.parquet", 1);
        let dv_b = dv("d/dv.puffin", 44, "d/b.parquet", 1);
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![dv_a.clone(), dv_b])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![dv("d/dv2.puffin", 4, "d/a.parquet", 2)])
            .remove_delete_files(vec![dv_a])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let mut live: Vec<_> = entries(&table)
            .await
            .into_iter()
            .filter(|e| e.0 == ManifestContentType::Deletes && e.1 != ManifestStatus::Deleted)
            .map(|e| (e.2, e.3))
            .collect();
        live.sort();
        assert_eq!(live, vec![
            ("d/dv.puffin".to_string(), Some(44)),
            ("d/dv2.puffin".to_string(), Some(4)),
        ]);
    }

    #[tokio::test]
    async fn test_rejects_removing_a_delete_file_that_is_not_live() {
        let catalog = new_memory_catalog().await;
        let table = table_in_catalog(&catalog, FormatVersion::V3).await;
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("d/a.parquet")])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let tx = Transaction::new(&table);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![dv("d/dv2.puffin", 4, "d/a.parquet", 1)])
            .remove_delete_files(vec![dv("d/missing.puffin", 4, "d/a.parquet", 1)])
            .apply(tx)
            .unwrap();
        assert_eq!(
            tx.commit(&catalog).await.unwrap_err().kind(),
            ErrorKind::DataInvalid
        );
    }

    #[tokio::test]
    async fn test_position_delete_formats_follow_the_format_version() {
        let catalog = new_memory_catalog().await;
        let v3 = table_in_catalog(&catalog, FormatVersion::V3).await;
        let tx = Transaction::new(&v3);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![position_delete_file("d/pos.parquet")])
            .apply(tx)
            .unwrap();
        assert_eq!(
            tx.commit(&catalog).await.unwrap_err().kind(),
            ErrorKind::DataInvalid
        );

        let v2 = table_in_catalog(&catalog, FormatVersion::V2).await;
        let tx = Transaction::new(&v2);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![dv("d/dv.puffin", 4, "d/a.parquet", 1)])
            .apply(tx)
            .unwrap();
        assert_eq!(
            tx.commit(&catalog).await.unwrap_err().kind(),
            ErrorKind::DataInvalid
        );

        let tx = Transaction::new(&v2);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file("d/a.parquet")])
            .apply(tx)
            .unwrap();
        let v2 = tx.commit(&catalog).await.unwrap();
        let tx = Transaction::new(&v2);
        let tx = tx
            .row_delta()
            .add_delete_files(vec![position_delete_file("d/pos.parquet")])
            .apply(tx)
            .unwrap();
        let v2 = tx.commit(&catalog).await.unwrap();
        let tasks: Vec<_> = v2
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(tasks[0].deletes().len(), 1);
        assert_eq!(tasks[0].deletes()[0].file_path(), "d/pos.parquet");
    }

    /// Removing a second data file from a manifest an earlier row delta rewrote must not bring
    /// back the file that rewrite marked deleted.
    #[tokio::test]
    async fn test_rewriting_a_rewritten_manifest_keeps_deleted_files_deleted() {
        let catalog = new_memory_catalog().await;
        let table = table_in_catalog(&catalog, FormatVersion::V2).await;
        let (a, b, c) = (
            data_file("d/a.parquet"),
            data_file("d/b.parquet"),
            data_file("d/c.parquet"),
        );
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![a.clone(), b.clone(), c])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let tx = Transaction::new(&table);
        let tx = tx.row_delta().remove_data_files(vec![a]).apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let tx = Transaction::new(&table);
        let tx = tx.row_delta().remove_data_files(vec![b]).apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tasks: Vec<_> = table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let paths: Vec<_> = tasks
            .iter()
            .map(|t| t.data_file_path().to_string())
            .collect();
        assert_eq!(paths, vec!["d/c.parquet".to_string()]);
    }
}
