# Selected AWS environment

| Setting | Selected value | Verification state |
| --- | --- | --- |
| Provider | AWS EC2 | Decision recorded; provision not verified |
| Region | `ap-south-1` (Mumbai) | Decision recorded |
| Instance | `t3.large` | Decision recorded; provision not verified |
| vCPUs | 2 | Instance specification |
| Memory | 8 GiB | Instance specification |
| Accelerator | No GPU | CPU-only by design |
| Operating system | Ubuntu Server 24.04 LTS | Intended AMI/OS; not inspected |
| Architecture | x86-64 | Chosen platform |
| Storage | 30 GiB gp3 EBS | Intended provision; not inspected |
| Payment | On-Demand | Chosen approach |

**Operational cautions:** T3 is credit-based. Sustained CPU usage may consume credits and can incur surplus-credit costs depending on mode; storage and public IPv4 may have separate costs. Stop the instance when not in use; EBS typically continues to incur storage charges. There is no requirement for public web-serving ports. Do not put secrets in project files or output transcripts.

CPU-only is a deliberate learning choice: small-scale transparency matters more than throughput. Do not add CUDA, GPU drivers, distributed model execution, model quantization, or deployment machinery for Phase 1.
