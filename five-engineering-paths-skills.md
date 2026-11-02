# Five engineering paths — basic and intermediate skills

Linux, scripting, Git, testing, and debugging are worth learning across all five paths. What changes is the problem you investigate. A backend engineer asks why an order was created twice. An AI inference engineer asks why ten simultaneous users wait too long for the first token. A VLSI CAD engineer asks why a chip design stopped meeting its timing requirement.

This guide uses one representative job for each path. **Basic** means you can complete a routine task using documentation. **Intermediate** means you can handle a failure or a changed requirement and demonstrate that your solution works.

## Skills shared by all five roles

| Shared skill | Basic working ability | Intermediate working ability |
|---|---|---|
| Linux and the shell | Navigate files, search logs, use pipes and environment variables, run a program, and check its exit code. | Diagnose why a program works on one machine but fails on another. Check permissions, dependencies, processes, memory, and disk. |
| Programming and scripting | Write a small program that reads input, processes it, and produces output. Use functions, lists, dictionaries, and error handling. | Automate a repeated task across many inputs. Handle partial failures and report which items succeeded, failed, or need another attempt. |
| Git and code review | Read a diff, make a focused commit, open a pull request, and review a small change. | Trace a regression to a change, revert it safely, and add a test that would have caught it. |
| Testing and debugging | Define the expected result, reproduce a failure, and check the actual result. | Isolate the failing stage, test one explanation at a time, and check the fix against both failing and previously working cases. |
| Configuration and reproducibility | Identify a program’s inputs, configuration, dependencies, and outputs. | Reproduce a teammate’s run using the same versions and settings. Explain why two runs produced different results. |
| Measurement and communication | Record what you ran, what happened, and what you expected. | Compare against a baseline and give a teammate enough evidence to reproduce the result or make a decision. |

**Language choice.** Python plus basic Bash gives you a way to practise the shared skills through API calls, automation, tests, benchmarks, and report processing. JavaScript or TypeScript also works well for managed AI and backend applications. For the VLSI path below, add Tcl because design tools use it to control flow stages. [OpenROAD Flow Scripts tutorial](https://openroad-flow-scripts.readthedocs.io/en/latest/tutorials/FlowTutorial.html)

## 1. Managed AI application engineer

**What you own.** A workflow that uses a hosted model or coding agent to complete a task. You define its instructions, the tools it can use, and the conditions for accepting its output.

- **Basic skills.** Call a hosted model from code, give it clear instructions and an output format, connect one tool, and validate the result. When using a coding agent, put project instructions in Markdown, inspect its diff, and run the relevant tests. [Building effective agents](https://www.anthropic.com/research/building-effective-agents)
- **Intermediate skills.** Build realistic test cases with expected outcomes. Trace a failed run to the instructions, tool choice, tool arguments, or application code. Measure whether a change improves task success and what it costs in latency or tokens. [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) · [Evaluating AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- **Example task.** A database assistant answers “list the tables” correctly but invents column names when the user asks a follow-up. Inspect the trace, fix how it uses `read_columns`, and test both questions again.

## 2. Self-hosted AI and inference engineer

**What you own.** The model-serving endpoint. If you also build an agent harness, the tool-calling and evaluation skills from Role 1 carry over. The additional responsibility is making the model run within hardware, latency, and cost limits.

- **Basic skills.** Understand tokens, model weights, quantization, context length, and the difference between system RAM and GPU memory. Start one model server, send a request, inspect its logs, and measure how long the first token takes to arrive. [vLLM serving benchmarks](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- **Intermediate skills.** Run a repeatable load test. Distinguish **time to first token**, **generation speed per user**, and **total throughput across users**. Investigate queueing, batching, and the memory used by the KV cache, which stores state from earlier tokens. [vLLM production metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
- **Example task.** Your database assistant responds quickly for one user but slowly for ten. Keep the model and requests fixed, measure queue time and generation speed separately, then change one serving setting and rerun the test.

## 3. VLSI CAD/EDA engineer

**What you own.** As a CAD flow automation engineer, you help chip designers run and debug the tools that turn a hardware description into a chip layout. Developing the algorithms inside a placement, routing, or simulation tool is a separate specialization.

- **Basic skills.** Learn digital logic, clocks, registers, and enough Verilog to read a small hardware design. Run it through synthesis, placement, and routing. Use Tcl to control tools and Python or shell scripts to collect logs and reports. [OpenROAD Flow Scripts tutorial](https://openroad-flow-scripts.readthedocs.io/en/latest/tutorials/FlowTutorial.html)
- **Intermediate skills.** Read timing reports and design constraints. Compare a new run with a known-good run across timing, area, runtime, and design-rule violations. Automate the comparison across multiple designs so an improvement in one design does not hide a regression in another. [OpenROAD flow metrics](https://openroad-flow-scripts.readthedocs.io/en/latest/contrib/Metrics.html)
- **Example task.** After a tool update, one block has **negative timing slack**, meaning it misses a timing requirement. Reproduce the old and new runs with the same inputs, locate the first stage where timing worsens, and show the design team the affected paths. [Tracking timing slack across flow steps](https://openroad-flow-scripts.readthedocs.io/en/latest/contrib/EndpointSlackTracking.html)

## 4. DevOps and cloud engineer

**What you own.** As a cloud infrastructure engineer, you deploy and operate application services.

- **Basic skills.** Understand IP addresses, DNS, ports, TLS, and permissions. Package an application in a container, deploy it through a pipeline, configure a health check, and find its logs. Learn one cloud provider’s compute, networking, and access controls.
- **Intermediate skills.** Define infrastructure in version-controlled code, make deployments repeatable, plan a rollback, and diagnose failures across the application, network, and infrastructure. Use monitoring to determine whether a release has affected users. [Google SRE on monitoring](https://sre.google/workbook/monitoring/) · [Incident response](https://sre.google/workbook/incident-response/) · [Configuration design](https://sre.google/workbook/configuration-design/)
- **Example task.** A release passes the build, but users receive `502` errors. Trace a request from the load balancer to the application. Check whether the new instances are healthy and listening on the expected port, then verify recovery after a fix or rollback.

## 5. Backend engineer progressing toward architecture

**What you own.** The service’s behavior, its data, and what happens when requests overlap, fail, or are retried.

- **Basic skills.** Build an HTTP endpoint, validate input, check permissions, design a database table, write SQL, and test success and failure cases. Follow one request from the handler through the database and back.
- **Intermediate skills.** Use transactions and constraints to protect data, read a query plan to diagnose a slow endpoint, and make retries safe. Practise **high-level design (HLD)** by deciding which work belongs in the API, database, or background worker. Practise **low-level design (LLD)** by defining interfaces, transaction boundaries, and error behavior. [PostgreSQL on query plans](https://www.postgresql.org/docs/current/using-explain.html) · [AWS on making retries safe](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
- **Example task.** A client times out after submitting an order. The database committed it, but the client missed the response and retries. Design the API and database operation so the retry returns the original order. Test two identical requests arriving simultaneously.

## First practice exercise

1. Choose a command-line program you already understand and run it on Linux.
2. Write a script that runs it against ten inputs, including one deliberately invalid input.
3. Record the result for each input and capture any error.
4. Add a test for the failure case and commit the change to Git.

**Completion criterion.** Another person can run one command, see exactly which input failed and why, fix it, and reproduce the successful result.
