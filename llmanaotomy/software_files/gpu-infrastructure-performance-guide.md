# GPU infrastructure and performance guide

## 1. Purpose and authority

Use this guide for accelerator-backed training, inference, and other GPU workloads.
It owns hardware and software compatibility, device allocation, resource behavior, and performance measurement.
Follow the [core workflow](software-ai-development-workflow.md), [architecture guide](architecture-and-design-guide.md), and [quality guide](engineering-quality-guide.md).
It does not authorize hardware purchases, paid compute, driver changes, or cluster administration.

## 2. Related guides

- [ML and deep learning](ml-deep-learning-guide.md): training correctness, experiments, and model evaluation.
- [LLM engineering](llm-engineering-guide.md): inference workloads, context lengths, quality, and application behavior.
- [Infrastructure](infrastructure-deployment-operations-guide.md): provisioning, containers, Kubernetes, identity, and operations.
- [Distributed systems](distributed-systems-microservices-guide.md): worker coordination and partial failure.

Model quality belongs to the model guides. Device throughput alone does not establish model quality.
Apply this guide only when accelerator execution is relevant.

## 3. Define the workload

Identify whether the workload trains, fine-tunes, performs batch inference, serves interactive requests, or runs another computation.
Record model or algorithm, data shape, precision, batch size, concurrency, and expected operating conditions.
For LLM inference, include input and output lengths and relevant cache behavior.
State latency, throughput, memory, reliability, and cost targets supplied by the project.
Do not infer a required GPU model or device count from a model name alone.

Separate peak capacity, sustained capacity, and acceptable user experience.
Record which resources and environments are authorized for experiments.

## 4. Compatibility and inventory

Identify actual devices, available memory, platform architecture, operating system, and relevant interconnects.
Record driver, accelerator runtime, framework, communication libraries, and compiled extensions where applicable.
Identify container or scheduler integrations that expose devices to workloads.

Verify version-sensitive compatibility with current vendor documentation during approved technical work.
Do not assume a newer component is compatible with every older component.
Distinguish host driver requirements from libraries packaged with the application.
Check that the workload uses the intended device and accelerated execution path.
An installed GPU package does not prove GPU execution.

## 5. HLD and LLD

At HLD level, describe compute nodes, device topology, storage, network, scheduling, and failure domains.
Identify shared-resource boundaries and operational ownership.
Explain single-device or distributed execution choices against actual workload requirements.

At LLD level, define:

- Device selection and placement.
- Memory budgeting and allocation behavior.
- Precision, kernels, and relevant compilation assumptions.
- Batching, concurrency, queue limits, and cancellation.
- Data loading and transfer behavior.
- Synchronization and inter-device communication.
- Checkpoint, restart, and error handling.
- Measurements and acceptance criteria.

## 6. Memory and resource planning

Estimate model parameters, optimizer state, gradients, activations, workspace, and caches as applicable.
Include host memory, pinned memory, storage, and transfer buffers where relevant.
State assumptions about inputs, batch sizes, sequence lengths, and concurrent work.
Validate estimates with representative measurements.

Account for allocator behavior, fragmentation, and temporary peaks where they affect capacity.
Define out-of-memory handling and resource cleanup.
Do not rely on unbounded retries with smaller batches without an approved fallback policy.
Record how a fallback changes throughput, latency, or numerical behavior.

## 7. Performance methodology

Define the benchmark question and workload before optimizing.
Record hardware, software, power or sharing conditions where relevant, and workload configuration.
Separate warm-up, compilation, loading, and steady-state measurements.
Use timing methods appropriate to asynchronous device execution.
Include enough repetitions or samples to support the comparison.

Measure user-relevant behavior, not only kernel speed.
For inference, distinguish time to first output, generation rate, end-to-end latency, and throughput when relevant.
For training, measure useful work per unit time alongside convergence and quality.
Report memory, cost, and bottlenecks with performance results.
Do not compare materially different workloads without explaining the differences.

## 8. Optimization

Investigate measured bottlenecks before changing the design.
Potential bottlenecks include data loading, storage, host processing, transfers, device compute, memory bandwidth, and communication.
Evaluate optimizations such as batching, precision changes, quantization, compilation, or parallelism only when appropriate.

For each material optimization, record:

1. Hypothesis and baseline.
2. Changed configuration or code.
3. Comparable workload and measurement method.
4. Performance and resource effects.
5. Correctness or model-quality effects.
6. Compatibility, maintenance, and recovery consequences.

Retain the simpler approach when complexity has no demonstrated benefit for the project.
Do not trade away required quality solely to improve a throughput number.

## 9. Multi-GPU and multi-node work

Define partitioning, synchronization, communication, and worker roles.
Record topology and network assumptions that affect performance or correctness.
Measure scaling efficiency rather than assuming linear gains from additional devices.
Identify collective-operation timeouts and worker-failure behavior where applicable.

Coordinate checkpoint consistency and restart semantics with the ML guide.
Test whether a partial worker failure leaves recoverable work or requires a complete restart.
Do not present resumed training as equivalent to uninterrupted training without supporting evidence.
Bound retries and preserve failure diagnostics.

## 10. Scheduling, isolation, and operation

Use the infrastructure guide for deployment and access controls.
Define device allocation, sharing, admission, and priority policies as project requirements dictate.
Verify the actual isolation provided by the selected platform.
Do not assume namespace or container separation alone provides every required accelerator isolation property.

Monitor utilization, memory, temperature or power conditions, errors, queue delay, and user-facing performance where relevant.
Define response to device loss, resource leaks, stalled work, and capacity exhaustion.
Coordinate driver and runtime upgrades with workload compatibility checks.
Preserve recoverable environment and artifact references.

## 11. Verification and reporting

Verify the intended execution path, numerical behavior, representative performance, and material recovery cases.
Use the model guides for quality reevaluation after precision, batching, quantization, or parallelism changes.
Record workload scope and uncertainty behind capacity claims.
Identify untested devices or configurations as limitations.
Do not claim that high utilization alone means efficient or correct operation.

Report achieved results against the project's actual targets.
Keep benchmarks, environment references, and design decisions recoverable through the core records.
Close work under the core workflow after applicable checks and required reviews.
