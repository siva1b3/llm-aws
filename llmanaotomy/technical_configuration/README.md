# 04 — Technical Configuration and Decisions

**Document role:** Authoritative infrastructure, model, software, and reproducibility choices  
**Update when:** EC2/model/dependency choices change, installed versions are observed, asset paths are known, or actual runtime settings are verified.

> **Important:** This file records *selected configuration*, not evidence that an EC2 instance exists, dependencies have been installed, or the model has run. Record observed state separately from intended state.

## Topic files

Read the relevant file below for the topic you need. Each file owns one section of this domain.

- [Selected AWS environment](selected_aws_ec2_environment.md)
- [Selected Hugging Face model](selected_hugging_face_model_and_rationale.md)
- [Minimal software decisions](software_dependencies_and_initial_runtime_choices.md)
- [Asset inventory: expected categories, not confirmed filenames](expected_model_assets_and_download_inventory.md)
- [Execution and asset lifecycle we will inspect](execution_and_asset_lifecycle.md)
- [Runtime record — fill only with observed facts](observed_environment_and_runtime_record.md)
- [Decision change protocol](technical_decision_change_protocol.md)

## Maintaining this folder

Edit the file that owns the affected topic. For a new topic, add a Markdown file with a descriptive snake_case filename and link it above. Keep existing filenames stable. Update [AGENTS.md](../AGENTS.md) only if routing or domain ownership changes.
