# 03 — Implementation, Inspection, and Verification Rules

**Document role:** Authoritative coding discipline, observability conventions, and completion gates  
**Update when:** Code/inspection conventions or the evidence required to verify a learning step changes.

## Topic files

Read the relevant file below for the topic you need. Each file owns one section of this domain.

- [Non-negotiable implementation discipline](incremental_implementation_and_dependency_discipline.md)
- [Phase-1 execution contract](single_forward_pass_execution_contract.md)
- [For every operation, inspect the data contract](operation_input_computation_and_output_inspection.md)
- [One preferred human-readable display per data type](human_readable_display_formats_by_data_type.md)
- [Avoid overwhelming output](bounded_tensor_and_terminal_output.md)
- [When to inspect model source code or reproduce an operation](model_source_inspection_and_computation_reproduction.md)
- [Verification gate: all conditions required](technical_verification_and_learner_confirmation_gate.md)
- [File-change and test discipline](file_changes_testing_and_evidence_records.md)
- [Error handling for teaching](minimal_error_diagnosis_and_correction.md)

## Maintaining this folder

Edit the file that owns the affected topic. For a new topic, add a Markdown file with a descriptive snake_case filename and link it above. Keep existing filenames stable. Update [AGENTS.md](../AGENTS.md) only if routing or domain ownership changes.
