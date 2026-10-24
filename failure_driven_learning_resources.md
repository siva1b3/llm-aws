# Failure-Driven Learning — Resource Catalog

**Edition:** 1.1  
**Research checked:** 24 October 2026  
**Scope:** AWS, Azure, GCP, Terraform, Linux, Docker, Kubernetes, networking, databases, backend systems, distributed systems, observability, SRE, postmortems and chaos engineering.  
**Status:** Resource discovery and categorization only. **No learning sequence or course plan is defined yet.**

> **Purpose:** Create a reference library of places to encounter failures, observe real incidents, reproduce faults in safe environments, debug them, understand contributing factors and verify a fix. An incident report is evidence to study; a chaos tool is a way to create a failure; a challenge platform is a place to solve one. These are different resources.

## Contents

1. [How to use this catalog](#1-how-to-use-this-catalog)
2. [Master directory](#2-master-directory)
3. [Resources by category](#3-resources-by-category)
4. [Browse by technical domain](#4-browse-by-technical-domain)
5. [Browse by failure mode](#5-browse-by-failure-mode)
6. [How to study reports and build safe reproductions](#6-how-to-study-reports-and-build-safe-reproductions)
7. [Verification, pricing and limitations](#7-verification-pricing-and-limitations)
8. [Future roadmap placeholder](#8-future-roadmap-placeholder)

**Revision note (v1.1):** Added KodeKloud Engineer, Playgrounds, Kubernetes troubleshooting labs and free hands-on labs; classification distinguishes scenario-based incidents from guided instruction. Total: **67 catalog entries**.

## 1. How to use this catalog

**Resource types**

| Type | What you actually get | What it is **not** |
|---|---|---|
| **Break/fix lab** | A partially or fully broken system plus a goal; you investigate and repair it. | A passive tutorial. |
| **Guided simulation** | A disposable environment with tasks, hints and often reference answers. | Necessarily a realistic outage. |
| **Game day** | A limited-time open-ended operational challenge. | Usually a permanent public sandbox. |
| **Fault-injection tool** | Software that injects controlled faults into an environment you own. | A collection of prebroken labs. |
| **Postmortem/incident report** | A real event with symptoms, timeline, contributing factors, response and follow-up. | A runnable exercise by itself. |
| **Incident index / blog** | Many reports or articles you can search. | A guarantee that every link has deep technical detail. |
| **Reference** | Investigation methods, architecture principles and recovery techniques. | Failure-first practice on its own. |

**Access labels:** **Free reading** = public page; **free subset** = selected labs, with paid plans; **mixed** = scenario-specific; **cloud usage** = you may pay for compute, networking, storage and/or experiments. Pricing and availability can change. **Reproduction labels:** **prebuilt**, **self-designed**, **partial**, **conceptual**, **none**.

**Selection principle:** Start cataloging all relevant resources now. Do *not* interpret their position in this document or ID numbers as a difficulty ranking or learning order.

## 2. Master directory

All entries have a stable ID so you can refer to them in later notes, an incident tracker or a learning plan. Duplicate references to the same underlying URL are explicitly marked in their detailed entries.

### LAB — Interactive break/fix and hands-on platforms (11)

| ID | Resource | Main coverage | Type |
|---|---|---|---|
| LAB-01 | [SadServers](https://sadservers.com/scenarios) | Linux, Network, Docker | Break/fix |
| LAB-02 | [Learn by Fixing](https://www.learnbyfixing.com/scenarios/) | Linux, Docker, Kubernetes | Break/fix |
| LAB-03 | [Killercoda](https://killercoda.com/about) | Linux, Kubernetes, Docker | Guided hands-on / some break-fix |
| LAB-04 | [Killercoda CKA Scenarios](https://killercoda.com/cka) | Kubernetes, Networking, Storage | Guided / troubleshoot |
| LAB-05 | [Killercoda CKS Scenarios](https://killercoda.com/cks) | Kubernetes, Security, Networking | Guided / troubleshoot |
| LAB-06 | [Killer Shell Simulators](https://killer.sh/) | Kubernetes, Linux, Security | Scenario-based assessment |
| LAB-07 | [AWS GameDay](https://aws.amazon.com/gameday/) | AWS, Cloud, DistributedSystems | Open-ended game day |
| LAB-08 | [KodeKloud Engineer](https://engineer.kodekloud.com/) | Linux, Docker, Kubernetes, DevOps, Cloud | Fictional workplace tickets / hands-on tasks |
| LAB-09 | [KodeKloud Playgrounds](https://kodekloud.com/playgrounds) | AWS, Terraform, Kubernetes, Linux, Kafka, Databases | Open sandbox / self-designed break-fix |
| LAB-10 | [KodeKloud Kubernetes Troubleshooting](https://kodekloud.com/courses/kubernetes-troubleshooting-for-application-developers) | Kubernetes, Network, Security, Storage | Scenario-based troubleshooting course |
| LAB-11 | [KodeKloud Free Hands-on Labs](https://kodekloud.com/studio/labs) | Terraform, Kubernetes, Linux, Docker, Git | Guided interactive labs |

#### LAB-08 — KodeKloud Engineer

**Link:** https://engineer.kodekloud.com/  
**Description:** A fictional company's systems and engineering assignments. You are given task briefs and work with real tooling in environments representing operational work. Some tasks focus on investigating failures and correcting configurations; others involve setup, deployment or administration.  
**Topics:** Linux, DevOps, Docker, Kubernetes, Cloud, CI/CD  
**Learning format:** Workplace-style tasks and troubleshooting; automated task validation  
**Access / environment:** Free basic access; paid tiers expand task access and attempt frequency; hosted/browser lab  
**Can I recreate a failure?** Prebuilt tasks; troubleshooting content varies by task  
**Limitations / cautions:** These are simulated company scenarios, not original incident records; not every task is a failure diagnosis. Free/standard access has task-unlock and daily attempt limits.

#### LAB-09 — KodeKloud Playgrounds

**Link:** https://kodekloud.com/playgrounds  
**Description:** Temporary environments for AWS/Azure/GCP, Linux, Kubernetes (including multi-node and Calico), Docker, Terraform/LocalStack, Kafka, PostgreSQL, Prometheus/Grafana, and others. You can intentionally create a fault, investigate and repair it without maintaining a complete local lab.  
**Topics:** AWS, Azure, GCP, Linux, Docker, Kubernetes, Terraform, Database, Kafka, Observability  
**Learning format:** Open sandbox / self-designed break-fix  
**Access / environment:** Some free access is limited; most playground capabilities require a paid plan; browser sandbox  
**Can I recreate a failure?** Yes, self-designed within the sandbox's permitted operations  
**Limitations / cautions:** Cloud playgrounds restrict permissions and supported services. Temporary environment lifetime and plan eligibility vary; they are not equivalent to unrestricted personal cloud accounts.

#### LAB-10 — KodeKloud Kubernetes Troubleshooting for Application Developers

**Link:** https://kodekloud.com/courses/kubernetes-troubleshooting-for-application-developers  
**Description:** A troubleshooting-focused course with lab scenarios involving image pull failures, crashing/pending pods, RBAC, network policies, ingress, inaccessible services, and storage multi-attach errors. More directly relevant than generic introduction courses.  
**Topics:** Kubernetes, Docker, Networking, RBAC, Storage  
**Learning format:** Guided explanations plus scenario-driven investigation labs  
**Access / environment:** Course/lab entitlement depends on KodeKloud plan; hosted lab  
**Can I recreate a failure?** Yes, scenario-specific guided labs  
**Limitations / cautions:** It includes video instruction and examples, not exclusively unexplained break/fix challenges; course availability and access should be confirmed before subscribing.

#### LAB-11 — KodeKloud Free Hands-on Labs

**Link:** https://kodekloud.com/studio/labs  
**Description:** Curated free interactive lab collections covering Terraform, Kubernetes, Git and DevOps fundamentals. Useful as a baseline when a learner needs a controlled environment before diagnosing advanced failures.  
**Topics:** Terraform, Kubernetes, Linux, DevOps, Git  
**Learning format:** Guided interactive exercises  
**Access / environment:** Selected labs free with an account; premium offerings require a subscription; browser lab  
**Can I recreate a failure?** Some exercises; most are guided tasks rather than initially broken systems  
**Limitations / cautions:** Not a replacement for incident-oriented debugging; check the exercise objective before choosing it.

### CLOUD — Cloud infrastructure labs and failure-testing services (10)

| ID | Resource | Main coverage | Type |
|---|---|---|---|
| CLOUD-01 | [AWS Fault Injection Service (FIS)](https://aws.amazon.com/fis/) | AWS, Cloud, Chaos | Inject-your-own failure |
| CLOUD-02 | [AWS FIS Stop Conditions](https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html) | AWS, Chaos, Observability | Reference / safety |
| CLOUD-03 | [AWS FIS Workshop Repository](https://github.com/aws-samples/aws-fault-injection-simulator-workshop) | AWS, Chaos, Cloud | Workshop / code |
| CLOUD-04 | [AWS Workshop Studio Catalog](https://catalog.workshops.aws/) | AWS, Cloud, Observability | Guided workshop directory |
| CLOUD-05 | [Terraform — Manage State](https://developer.hashicorp.com/terraform/tutorials/state) | Terraform, AWS, Cloud | Guided hands-on |
| CLOUD-06 | [Terraform — Manage Resource Drift](https://docs.hashicorp.com/terraform/tutorials/state/resource-drift) | Terraform, AWS, Cloud | Deliberate drift exercise |
| CLOUD-07 | [Terraform — Recover State](https://docs.hashicorp.com/terraform/cli/state/recover) | Terraform, Cloud | Reference / recovery |
| CLOUD-08 | [Azure Chaos Studio](https://learn.microsoft.com/en-us/azure/chaos-studio/chaos-studio-overview) | Azure, Cloud, Chaos | Inject-your-own failure |
| CLOUD-09 | [Azure Chaos Studio Scenario Catalog](https://learn.microsoft.com/azure/chaos-studio/chaos-studio-scenarios) | Azure, Chaos, Database | Template catalog |
| CLOUD-10 | [Google Cloud Incident Reports](https://status.cloud.google.com/) | GCP, Cloud, IncidentResponse | Read real incidents |

### CHAOS — Chaos-engineering tools and fault-injection frameworks (7)

| ID | Resource | Main coverage | Type |
|---|---|---|---|
| CHAOS-01 | [Chaos Mesh](https://chaos-mesh.org/docs/) | Kubernetes, Chaos, Networking | Inject-your-own failure |
| CHAOS-02 | [LitmusChaos](https://docs.litmuschaos.io/) | Kubernetes, Chaos, Observability | Inject-your-own failure |
| CHAOS-03 | [Litmus ChaosHub](https://hub.litmuschaos.io/) | Kubernetes, Chaos | Experiment templates |
| CHAOS-04 | [Chaos Toolkit](https://github.com/chaostoolkit/chaostoolkit) | Chaos, Cloud, DistributedSystems | Inject-your-own failure |
| CHAOS-05 | [Netflix Chaos Monkey](https://github.com/Netflix/chaosmonkey) | Cloud, Chaos, DistributedSystems | Inject-your-own failure |
| CHAOS-06 | [Netflix Simian Army (historical)](https://github.com/Netflix/SimianArmy) | Cloud, Chaos | Historical reference |
| CHAOS-07 | [Gremlin](https://www.gremlin.com/product) | Chaos, Cloud, SRE | Inject-your-own failure |

### COL — Postmortem collections and incident databases (7)

| ID | Resource | Main coverage | Type |
|---|---|---|---|
| COL-01 | [postmortem.io](https://postmortem.io/) | Postmortem, MultiTopic | Read and compare |
| COL-02 | [postmortems.app](https://postmortems.app/) | Postmortem, MultiTopic | Read / data exploration |
| COL-03 | [danluu/post-mortems](https://github.com/danluu/post-mortems) | Postmortem, MultiTopic | Browse index |
| COL-04 | [Kubernetes Failure Stories](https://k8s.af/) | Kubernetes, Networking, Postmortem | Browse curated reports |
| COL-05 | [GitHub Status / History](https://www.githubstatus.com/) | IncidentResponse, Cloud | Read status timelines |
| COL-06 | [Cloudflare Status](https://www.cloudflarestatus.com/) | IncidentResponse, Networking | Read status timelines |
| COL-07 | [AWS Post-Event Summaries](https://aws.amazon.com/premiumsupport/technology/pes/) | AWS, Cloud, Postmortem | Read reports |

### REAL — Original reports: real company incidents (11)

| ID | Resource | Main coverage | Type |
|---|---|---|---|
| REAL-01 | [AWS us-east-1 disruption — Dec 7, 2021](https://aws.amazon.com/message/12721//) | AWS, Networking, DistributedSystems | Original post-event summary |
| REAL-02 | [Cloudflare WAF regex/CPU outage — Jul 2, 2019](https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/) | Networking, Backend, Observability | Original technical postmortem |
| REAL-03 | [Cloudflare BGP configuration outage — Jun 21, 2022](https://blog.cloudflare.com/cloudflare-outage-on-june-21-2022/) | Networking, Cloud, IncidentResponse | Original technical postmortem |
| REAL-04 | [Cloudflare Bot Management data outage — Nov 18, 2025](https://blog.cloudflare.com/18-november-2025-outage/) | Database, Backend, Networking | Original technical postmortem |
| REAL-05 | [GitHub network partition / MySQL failover — Oct 2018](https://github.blog/news-insights/company-news/oct21-post-incident-analysis/) | Database, Networking, DistributedSystems | Original technical postmortem |
| REAL-06 | [GitLab database deletion — Jan 31, 2017](https://about.gitlab.com/blog/postmortem-of-database-outage-of-january-31/) | Database, Linux, IncidentResponse | Original technical postmortem |
| REAL-07 | [Atlassian multi-site deletion — Apr 2022](https://www.atlassian.com/blog/how-we-build/post-incident-review-april-2022-outage) | Cloud, Database, IncidentResponse | Original post-incident review |
| REAL-08 | [Meta/Facebook global backbone outage — Oct 4, 2021](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/) | Networking, DistributedSystems, IncidentResponse | Original technical report |
| REAL-09 | [Slack network saturation outage — Jan 4, 2021](https://slack.engineering/slacks-outage-on-january-4th-2021/) | Networking, Observability, Cloud | Original technical postmortem |
| REAL-10 | [Fastly global outage — Jun 8, 2021](https://fastly.quic.co/blog/summary-of-june-8-outage) | Networking, Backend, Cloud | Original company incident summary |
| REAL-11 | [Google Cloud multi-service outage — Jun 12, 2025](https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW) | GCP, Cloud, DistributedSystems | Official status / incident report |

### BLOG — Engineering blogs and official incident feeds (8)

| ID | Resource | Main coverage | Type |
|---|---|---|---|
| BLOG-01 | [Cloudflare Blog — Outage / Deep Dives](https://blog.cloudflare.com/) | Networking, Backend, Postmortem | Engineering blog |
| BLOG-02 | [GitHub Engineering Blog](https://github.blog/engineering/) | Database, Cloud, DistributedSystems | Engineering blog |
| BLOG-03 | [GitLab Engineering / Blog](https://about.gitlab.com/blog/) | Database, DevOps, IncidentResponse | Engineering blog |
| BLOG-04 | [Slack Engineering](https://slack.engineering/) | Backend, DistributedSystems, Observability | Engineering blog |
| BLOG-05 | [Meta Engineering](https://engineering.fb.com/) | Networking, DistributedSystems | Engineering blog |
| BLOG-06 | [Netflix TechBlog](https://netflixtechblog.com/) | Chaos, DistributedSystems, Backend | Engineering blog |
| BLOG-07 | [AWS Builders Library](https://builder.aws.com/learn/topics/builders-library) | AWS, DistributedSystems, SRE | Engineering practices |
| BLOG-08 | [Google Cloud Service Health](https://status.cloud.google.com/) | GCP, Cloud, Postmortem | Incident feed |

### GIT — Open-source GitHub repositories and exercise collections (7)

| ID | Resource | Main coverage | Type |
|---|---|---|---|
| GIT-01 | [bregman-arie/devops-exercises](https://github.com/bregman-arie/devops-exercises) | Linux, AWS, Docker | Exercises / questions |
| GIT-02 | [chaos-mesh/chaos-mesh](https://github.com/chaos-mesh/chaos-mesh) | Kubernetes, Chaos | Tool source |
| GIT-03 | [litmuschaos/litmus](https://github.com/litmuschaos/litmus) | Kubernetes, Chaos | Tool source |
| GIT-04 | [microsoft/chaos-studio](https://github.com/microsoft/chaos-studio) | Azure, Chaos, Cloud | Samples / scenario code |
| GIT-05 | [danluu/post-mortems](https://github.com/danluu/post-mortems) | Postmortem, MultiTopic | Dataset / link index |
| GIT-06 | [icco/postmortems](https://github.com/icco/postmortems) | Postmortem, MultiTopic | Dataset / structured metadata |
| GIT-07 | [AWS FIS Workshop Source](https://github.com/aws-samples/aws-fault-injection-simulator-workshop) | AWS, Chaos | Workshop repository |

### REF — SRE method, troubleshooting and postmortem references (6)

| ID | Resource | Main coverage | Type |
|---|---|---|---|
| REF-01 | [Google SRE Book](https://sre.google/sre-book/table-of-contents/) | SRE, DistributedSystems, Observability | Reference book |
| REF-02 | [Google SRE Workbook — Postmortem Culture](https://sre.google/workbook/postmortem-culture/) | SRE, IncidentResponse, Postmortem | Methods / templates |
| REF-03 | [Google SRE — Incident Management Guide](https://sre.google/resources/practices-and-processes/incident-management-guide/) | SRE, IncidentResponse | Runbook / method |
| REF-04 | [Google SRE — Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/) | DistributedSystems, Backend, Observability | Engineering reference |
| REF-05 | [AWS FIS Stop Conditions](https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html) | AWS, Chaos, Safety | Safety reference |
| REF-06 | [Terraform State Recovery Guide](https://docs.hashicorp.com/terraform/cli/state/recover) | Terraform, Cloud | Recovery reference |

## 3. Resources by category

### LAB — Interactive break/fix platforms

Preconfigured environments that present a technical problem. Some are true troubleshooting challenges; others are guided tasks or simulations.

#### LAB-01 — SadServers

**Link:** https://sadservers.com/scenarios  
**Description:** Intentionally broken Linux and DevOps VMs. Scenarios span DNS, ports, systemd, databases, Docker, Kubernetes, Terraform and observability, with a check-my-solution mechanism.  
**Topics:** Linux, Network, Docker, Kubernetes, Database, Terraform, Observability  
**Learning format:** Break/fix  
**Access / environment:** Free subset; paid tiers; Browser VM  
**Can I recreate a failure?** Yes, prebuilt  
**Limitations / cautions:** Some labs require registration or a paid plan; some also ask you to configure rather than fix.

#### LAB-02 — Learn by Fixing

**Link:** https://www.learnbyfixing.com/scenarios/  
**Description:** Short incident-style Linux, Docker and Kubernetes tasks such as a port conflict, failed Docker Compose stack or pending pod. Offers several free, no-signup scenarios.  
**Topics:** Linux, Docker, Kubernetes, Database  
**Learning format:** Break/fix  
**Access / environment:** Free subset; verify access per scenario; Browser lab  
**Can I recreate a failure?** Yes, prebuilt  
**Limitations / cautions:** Catalog contains both troubleshooting and optimization/configuration tasks.

#### LAB-03 — Killercoda

**Link:** https://killercoda.com/about  
**Description:** Disposable browser-based Linux and Kubernetes environments. Community-authored scenarios teach concrete operational tasks, including CKA/CKS material.  
**Topics:** Linux, Kubernetes, Docker, Networking, Security  
**Learning format:** Guided hands-on / some break-fix  
**Access / environment:** Free scenarios; PLUS and some courses paid; Browser VM / cluster  
**Can I recreate a failure?** Yes, scenario-dependent  
**Limitations / cautions:** Not every scenario starts broken; some are tutorials.

#### LAB-04 — Killercoda CKA Scenarios

**Link:** https://killercoda.com/cka  
**Description:** Kubernetes administration task environments including cluster operations and troubleshooting. Complements general Killercoda by collecting CKA-oriented scenarios.  
**Topics:** Kubernetes, Networking, Storage  
**Learning format:** Guided / troubleshoot  
**Access / environment:** Mixed; community and course access differ; Browser cluster  
**Can I recreate a failure?** Yes, scenario-dependent  
**Limitations / cautions:** Certification coverage is not the same as production incident response.

#### LAB-05 — Killercoda CKS Scenarios

**Link:** https://killercoda.com/cks  
**Description:** Hands-on Kubernetes security scenarios spanning RBAC, NetworkPolicy, admission, runtime detection and security configuration.  
**Topics:** Kubernetes, Security, Networking  
**Learning format:** Guided / troubleshoot  
**Access / environment:** Mixed; some features paid; Browser cluster  
**Can I recreate a failure?** Yes, scenario-dependent  
**Limitations / cautions:** Security certification tasks, not exclusively failure investigation.

#### LAB-06 — Killer Shell Simulators

**Link:** https://killer.sh/  
**Description:** Timed, exam-like environments for CKA, CKS, CKAD, Linux and platform engineering with solutions and scoring.  
**Topics:** Kubernetes, Linux, Security  
**Learning format:** Scenario-based assessment  
**Access / environment:** Paid or included with eligible exam purchases; Hosted simulator  
**Can I recreate a failure?** Yes, exam tasks  
**Limitations / cautions:** Exam simulations prioritize speed and task completion, not detailed postmortems.

#### LAB-07 — AWS GameDay

**Link:** https://aws.amazon.com/gameday/  
**Description:** Team-based, open-ended AWS challenge events with little step-by-step guidance. Participants solve realistic operational and architectural problems in event-provided environments.  
**Topics:** AWS, Cloud, DistributedSystems, IncidentResponse  
**Learning format:** Open-ended game day  
**Access / environment:** Free to participate in scheduled events; AWS event environment  
**Can I recreate a failure?** Yes, during events  
**Limitations / cautions:** Registration/event availability required; not a continuously available sandbox.

#### LAB-08 — KodeKloud Engineer

**Link:** https://engineer.kodekloud.com/  
**Description:** A fictional company's systems and engineering assignments. You are given task briefs and work with real tooling in environments representing operational work. Some tasks focus on investigating failures and correcting configurations; others involve setup, deployment or administration.  
**Topics:** Linux, DevOps, Docker, Kubernetes, Cloud, CI/CD  
**Learning format:** Workplace-style tasks and troubleshooting; automated task validation  
**Access / environment:** Free basic access; paid tiers expand task access and attempt frequency; hosted/browser lab  
**Can I recreate a failure?** Prebuilt tasks; troubleshooting content varies by task  
**Limitations / cautions:** These are simulated company scenarios, not original incident records; not every task is a failure diagnosis. Free/standard access has task-unlock and daily attempt limits.

#### LAB-09 — KodeKloud Playgrounds

**Link:** https://kodekloud.com/playgrounds  
**Description:** Temporary environments for AWS/Azure/GCP, Linux, Kubernetes (including multi-node and Calico), Docker, Terraform/LocalStack, Kafka, PostgreSQL, Prometheus/Grafana, and others. You can intentionally create a fault, investigate and repair it without maintaining a complete local lab.  
**Topics:** AWS, Azure, GCP, Linux, Docker, Kubernetes, Terraform, Database, Kafka, Observability  
**Learning format:** Open sandbox / self-designed break-fix  
**Access / environment:** Some free access is limited; most playground capabilities require a paid plan; browser sandbox  
**Can I recreate a failure?** Yes, self-designed within the sandbox's permitted operations  
**Limitations / cautions:** Cloud playgrounds restrict permissions and supported services. Temporary environment lifetime and plan eligibility vary; they are not equivalent to unrestricted personal cloud accounts.

#### LAB-10 — KodeKloud Kubernetes Troubleshooting for Application Developers

**Link:** https://kodekloud.com/courses/kubernetes-troubleshooting-for-application-developers  
**Description:** A troubleshooting-focused course with lab scenarios involving image pull failures, crashing/pending pods, RBAC, network policies, ingress, inaccessible services, and storage multi-attach errors. More directly relevant than generic introduction courses.  
**Topics:** Kubernetes, Docker, Networking, RBAC, Storage  
**Learning format:** Guided explanations plus scenario-driven investigation labs  
**Access / environment:** Course/lab entitlement depends on KodeKloud plan; hosted lab  
**Can I recreate a failure?** Yes, scenario-specific guided labs  
**Limitations / cautions:** It includes video instruction and examples, not exclusively unexplained break/fix challenges; course availability and access should be confirmed before subscribing.

#### LAB-11 — KodeKloud Free Hands-on Labs

**Link:** https://kodekloud.com/studio/labs  
**Description:** Curated free interactive lab collections covering Terraform, Kubernetes, Git and DevOps fundamentals. Useful as a baseline when a learner needs a controlled environment before diagnosing advanced failures.  
**Topics:** Terraform, Kubernetes, Linux, DevOps, Git  
**Learning format:** Guided interactive exercises  
**Access / environment:** Selected labs free with an account; premium offerings require a subscription; browser lab  
**Can I recreate a failure?** Some exercises; most are guided tasks rather than initially broken systems  
**Limitations / cautions:** Not a replacement for incident-oriented debugging; check the exercise objective before choosing it.

### CLOUD — Cloud infrastructure labs and failure-testing services

Cloud-native exercises and tooling, including infrastructure-as-code state problems. These can create billable resources.

#### CLOUD-01 — AWS Fault Injection Service (FIS)

**Link:** https://aws.amazon.com/fis/  
**Description:** Managed AWS fault injection service for resilience tests against cloud workloads. Define experiments, monitor impact, and verify whether workloads recover.  
**Topics:** AWS, Cloud, Chaos, Observability  
**Learning format:** Inject-your-own failure  
**Access / environment:** AWS usage charges may apply; Own AWS account  
**Can I recreate a failure?** Yes, self-designed  
**Limitations / cautions:** Use isolated resources, resource selection guardrails and stop conditions; not a zero-cost puzzle site.

#### CLOUD-02 — AWS FIS Stop Conditions

**Link:** https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html  
**Description:** Practical guardrail reference showing how a CloudWatch alarm can terminate an experiment when the system crosses a defined threshold.  
**Topics:** AWS, Chaos, Observability  
**Learning format:** Reference / safety  
**Access / environment:** Free documentation; experiments may cost; Own AWS account  
**Can I recreate a failure?** Supporting material  
**Limitations / cautions:** Set stop conditions before experiments and keep tests out of production initially.

#### CLOUD-03 — AWS FIS Workshop Repository

**Link:** https://github.com/aws-samples/aws-fault-injection-simulator-workshop  
**Description:** AWS-maintained sample workshop materials for learning fault injection, experiments and associated infrastructure.  
**Topics:** AWS, Chaos, Cloud  
**Learning format:** Workshop / code  
**Access / environment:** Free source; AWS resources may cost; GitHub + own AWS account  
**Can I recreate a failure?** Yes, with setup  
**Limitations / cautions:** Verify workshop freshness and read cleanup instructions before deployment.

#### CLOUD-04 — AWS Workshop Studio Catalog

**Link:** https://catalog.workshops.aws/  
**Description:** Central catalog for AWS hands-on workshops; useful for finding resilience, observability and fault injection exercises.  
**Topics:** AWS, Cloud, Observability, Chaos  
**Learning format:** Guided workshop directory  
**Access / environment:** Catalog free; individual workshop accounts/resources vary; Browser + possibly AWS  
**Can I recreate a failure?** Depends on workshop  
**Limitations / cautions:** Catalog is broader than failure-driven learning; curate relevant workshops.

#### CLOUD-05 — Terraform — Manage State

**Link:** https://developer.hashicorp.com/terraform/tutorials/state  
**Description:** Official HashiCorp state tutorials on importing resources, moving state and reconciling tracked infrastructure. Valuable after an accidental deletion of local IaC/state files.  
**Topics:** Terraform, AWS, Cloud  
**Learning format:** Guided hands-on  
**Access / environment:** Free docs; possible cloud charges; Local CLI + cloud  
**Can I recreate a failure?** Yes, scripted exercises  
**Limitations / cautions:** Apply/destroy exercises can create billable cloud resources.

#### CLOUD-06 — Terraform — Manage Resource Drift

**Link:** https://docs.hashicorp.com/terraform/tutorials/state/resource-drift  
**Description:** Deliberately change infrastructure outside Terraform, detect drift and reconcile the discrepancy. A focused way to study state versus actual infrastructure.  
**Topics:** Terraform, AWS, Cloud  
**Learning format:** Deliberate drift exercise  
**Access / environment:** Free docs; AWS resources may cost; Terraform CLI + AWS  
**Can I recreate a failure?** Yes, step-by-step  
**Limitations / cautions:** Avoid experimenting on valuable Terraform state or production resources.

#### CLOUD-07 — Terraform — Recover State

**Link:** https://docs.hashicorp.com/terraform/cli/state/recover  
**Description:** Official reference for disaster recovery of Terraform state using backups and backend operations.  
**Topics:** Terraform, Cloud  
**Learning format:** Reference / recovery  
**Access / environment:** Free documentation; Local CLI / remote backend  
**Can I recreate a failure?** Conceptual / reconstruct  
**Limitations / cautions:** State push, unlock and imports can have serious consequences if done against the wrong workspace.

#### CLOUD-08 — Azure Chaos Studio

**Link:** https://learn.microsoft.com/en-us/azure/chaos-studio/chaos-studio-overview  
**Description:** Azure-managed chaos and resilience testing. Its Workspaces/Scenarios model can reproduce DNS failures, compute disruption and dependency outages.  
**Topics:** Azure, Cloud, Chaos, Networking  
**Learning format:** Inject-your-own failure  
**Access / environment:** Azure service/resource charges may apply; Own Azure subscription  
**Can I recreate a failure?** Yes, self-designed  
**Limitations / cautions:** Workspaces/Scenarios are public preview as of verification date; review availability and safety.

#### CLOUD-09 — Azure Chaos Studio Scenario Catalog

**Link:** https://learn.microsoft.com/azure/chaos-studio/chaos-studio-scenarios  
**Description:** Concrete outage templates including DNS outages, zone-related failures, VM disruptions, cache issues and database failovers.  
**Topics:** Azure, Chaos, Database, Networking  
**Learning format:** Template catalog  
**Access / environment:** Free docs; Azure resource charges possible; Own Azure subscription  
**Can I recreate a failure?** Yes, templates  
**Limitations / cautions:** Templates only apply to supported resources and regions; many features remain preview.

#### CLOUD-10 — Google Cloud Incident Reports

**Link:** https://status.cloud.google.com/  
**Description:** Public GCP incident history and service-health details, including timelines and post-incident information where published.  
**Topics:** GCP, Cloud, IncidentResponse  
**Learning format:** Read real incidents  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Manual reproduction only  
**Limitations / cautions:** A status log is not always a deep engineering postmortem; individual reports vary.

### CHAOS — Chaos-engineering tools and fault-injection frameworks

Tools to inject controlled failures into YOUR own lab. These are not necessarily ready-made puzzles.

#### CHAOS-01 — Chaos Mesh

**Link:** https://chaos-mesh.org/docs/  
**Description:** Open-source Kubernetes-native chaos platform with Pod, network, DNS, HTTP, I/O and stress faults. Supports workflows and a dashboard.  
**Topics:** Kubernetes, Chaos, Networking, DistributedSystems  
**Learning format:** Inject-your-own failure  
**Access / environment:** Open source; hosting costs possible; Own Kubernetes cluster  
**Can I recreate a failure?** Yes  
**Limitations / cautions:** Requires cluster privileges; begin in an isolated test cluster.

#### CHAOS-02 — LitmusChaos

**Link:** https://docs.litmuschaos.io/  
**Description:** Open-source chaos framework for building and orchestrating experiments, including probes, workflows, dashboards and reusable experiment libraries.  
**Topics:** Kubernetes, Chaos, Observability  
**Learning format:** Inject-your-own failure  
**Access / environment:** Open source; hosting costs possible; Own Kubernetes cluster  
**Can I recreate a failure?** Yes  
**Limitations / cautions:** More setup than a prebuilt troubleshooting exercise.

#### CHAOS-03 — Litmus ChaosHub

**Link:** https://hub.litmuschaos.io/  
**Description:** Catalog of experiment definitions for controlled fault injection. Browse patterns rather than inventing every scenario from scratch.  
**Topics:** Kubernetes, Chaos  
**Learning format:** Experiment templates  
**Access / environment:** Open-source ecosystem; Own cluster / Litmus  
**Can I recreate a failure?** Yes, with setup  
**Limitations / cautions:** Validate experiment parameters, blast radius and compatibility with your cluster.

#### CHAOS-04 — Chaos Toolkit

**Link:** https://github.com/chaostoolkit/chaostoolkit  
**Description:** CLI-driven, extensible experiments for steady-state validation, fault methods and rollbacks across many targets. Good for infrastructure-as-code-style resilience tests.  
**Topics:** Chaos, Cloud, DistributedSystems  
**Learning format:** Inject-your-own failure  
**Access / environment:** Open source; target resources may cost; Local CLI + target system  
**Can I recreate a failure?** Yes  
**Limitations / cautions:** You design experiments; safe undo and permissions require care.

#### CHAOS-05 — Netflix Chaos Monkey

**Link:** https://github.com/Netflix/chaosmonkey  
**Description:** Open-source tool that terminates compute instances to test service resiliency. Historically important for learning automated failure injection.  
**Topics:** Cloud, Chaos, DistributedSystems  
**Learning format:** Inject-your-own failure  
**Access / environment:** Open source; infra may cost; Spinnaker-managed environment  
**Can I recreate a failure?** Yes, setup-heavy  
**Limitations / cautions:** The linked version requires Spinnaker; not the simplest lab starting point.

#### CHAOS-06 — Netflix Simian Army (historical)

**Link:** https://github.com/Netflix/SimianArmy  
**Description:** Historic suite that popularized intentionally breaking infrastructure to exercise reliability. Useful for understanding the origins of chaos engineering.  
**Topics:** Cloud, Chaos  
**Learning format:** Historical reference  
**Access / environment:** Free source; Read-only/historical  
**Can I recreate a failure?** Not recommended for new deployment  
**Limitations / cautions:** Repository explicitly states it is retired.

#### CHAOS-07 — Gremlin

**Link:** https://www.gremlin.com/product  
**Description:** Commercial resilience testing product with predefined reliability tests, scenario construction and infrastructure/application fault injection.  
**Topics:** Chaos, Cloud, SRE  
**Learning format:** Inject-your-own failure  
**Access / environment:** Commercial; trial advertised; Managed service / own targets  
**Can I recreate a failure?** Yes, on targets  
**Limitations / cautions:** Check current terms, pricing and blast-radius controls before connecting resources.

### COL — Postmortem collections and incident databases

Directories to browse many real incidents. The items here are indexes, not necessarily reports authored by the affected companies.

#### COL-01 — postmortem.io

**Link:** https://postmortem.io/  
**Description:** Searchable library of public incident reports with companies and technical tags (database, AWS, DNS, Kubernetes and more).  
**Topics:** Postmortem, MultiTopic  
**Learning format:** Read and compare  
**Access / environment:** Free browsing; Web  
**Can I recreate a failure?** Usually no  
**Limitations / cautions:** Aggregator; prefer linked primary-source reports when checking facts.

#### COL-02 — postmortems.app

**Link:** https://postmortems.app/  
**Description:** Annotated, category-tagged database of public postmortems, including automation, cloud, cascading failures and configuration changes.  
**Topics:** Postmortem, MultiTopic  
**Learning format:** Read / data exploration  
**Access / environment:** Free browsing; structured metadata; Web / JSON data  
**Can I recreate a failure?** Usually no  
**Limitations / cautions:** Community-maintained metadata; verify conclusions against original incidents.

#### COL-03 — danluu/post-mortems

**Link:** https://github.com/danluu/post-mortems  
**Description:** Large GitHub index of engineering postmortems organized around configuration errors, database problems, hardware, time and other failure classes.  
**Topics:** Postmortem, MultiTopic  
**Learning format:** Browse index  
**Access / environment:** Free; GitHub  
**Can I recreate a failure?** Usually no  
**Limitations / cautions:** Links may age or go offline; consult original sources.

#### COL-04 — Kubernetes Failure Stories

**Link:** https://k8s.af/  
**Description:** Curated incident and failure stories involving Kubernetes DNS, CNI, controllers, upgrades, memory pressure and control-plane issues.  
**Topics:** Kubernetes, Networking, Postmortem  
**Learning format:** Browse curated reports  
**Access / environment:** Free; Web  
**Can I recreate a failure?** Manual recreation  
**Limitations / cautions:** Some links point to historical talks/slides; check date and Kubernetes version.

#### COL-05 — GitHub Status / History

**Link:** https://www.githubstatus.com/  
**Description:** GitHub public service status and incident history. Useful for seeing user-facing symptoms and response timelines.  
**Topics:** IncidentResponse, Cloud  
**Learning format:** Read status timelines  
**Access / environment:** Free; Web  
**Can I recreate a failure?** No ready-made lab  
**Limitations / cautions:** Status updates may not include full engineering root-cause analysis.

#### COL-06 — Cloudflare Status

**Link:** https://www.cloudflarestatus.com/  
**Description:** Cloudflare current service health and history. Compare status signals with the company’s deeper engineering write-ups.  
**Topics:** IncidentResponse, Networking  
**Learning format:** Read status timelines  
**Access / environment:** Free; Web  
**Can I recreate a failure?** No ready-made lab  
**Limitations / cautions:** Different from a detailed postmortem; coverage varies by event.

#### COL-07 — AWS Post-Event Summaries

**Link:** https://aws.amazon.com/premiumsupport/technology/pes/  
**Description:** AWS portal for notable public post-event summaries describing service disruptions, contributing factors and mitigations.  
**Topics:** AWS, Cloud, Postmortem  
**Learning format:** Read reports  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Manual recreation  
**Limitations / cautions:** Not every operational event receives a public technical analysis.

### REAL — Original reports: real company incidents

Original first-party reports describing outages, contributing conditions, response and lessons.

#### REAL-01 — AWS us-east-1 disruption — Dec 7, 2021

**Link:** https://aws.amazon.com/message/12721//  
**Description:** A disruption involving AWS internal networking and the operational/control services that depend on it. Useful for understanding retries, shared dependencies and regional blast radius.  
**Topics:** AWS, Networking, DistributedSystems  
**Learning format:** Original post-event summary  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Conceptual / simulation  
**Limitations / cautions:** Do not attempt to reproduce regional-scale failure in an account.

#### REAL-02 — Cloudflare WAF regex/CPU outage — Jul 2, 2019

**Link:** https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/  
**Description:** A bad regular expression in a WAF rule caused severe CPU exhaustion and widespread errors. Strong study of software complexity, guardrails and staged rollout.  
**Topics:** Networking, Backend, Observability  
**Learning format:** Original technical postmortem  
**Access / environment:** Free reading; Web / local regex test  
**Can I recreate a failure?** Partial local demo possible  
**Limitations / cautions:** Recreate catastrophic backtracking only with bounded, isolated test input.

#### REAL-03 — Cloudflare BGP configuration outage — Jun 21, 2022

**Link:** https://blog.cloudflare.com/cloudflare-outage-on-june-21-2022/  
**Description:** A network configuration change withdrew critical routes from multiple data centers. Covers blast-radius control, rollout sequencing and rollback access.  
**Topics:** Networking, Cloud, IncidentResponse  
**Learning format:** Original technical postmortem  
**Access / environment:** Free reading; Web / optional network simulator  
**Can I recreate a failure?** Conceptual / advanced  
**Limitations / cautions:** Requires substantial routing knowledge for faithful reproduction.

#### REAL-04 — Cloudflare Bot Management data outage — Nov 18, 2025

**Link:** https://blog.cloudflare.com/18-november-2025-outage/  
**Description:** A database permission change altered generated feature data and disrupted traffic handling. A modern example of data/configuration propagation becoming a large-scale dependency failure.  
**Topics:** Database, Backend, Networking  
**Learning format:** Original technical postmortem  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Conceptual / partial  
**Limitations / cautions:** Reproduction would need a simplified mock pipeline, not real edge infrastructure.

#### REAL-05 — GitHub network partition / MySQL failover — Oct 2018

**Link:** https://github.blog/news-insights/company-news/oct21-post-incident-analysis/  
**Description:** Connectivity loss triggered database failover and divergent replication histories, forcing a difficult recovery prioritizing data integrity over availability.  
**Topics:** Database, Networking, DistributedSystems  
**Learning format:** Original technical postmortem  
**Access / environment:** Free reading; Web / MySQL lab  
**Can I recreate a failure?** Partial local simulation  
**Limitations / cautions:** A complete reproduction needs multi-site database topology.

#### REAL-06 — GitLab database deletion — Jan 31, 2017

**Link:** https://about.gitlab.com/blog/postmortem-of-database-outage-of-january-31/  
**Description:** Accidental removal of primary database data during troubleshooting caused major downtime and data loss. Highlights operator interfaces, backup validation and recoverability.  
**Topics:** Database, Linux, IncidentResponse  
**Learning format:** Original technical postmortem  
**Access / environment:** Free reading; Web / disposable PostgreSQL lab  
**Can I recreate a failure?** Partial local simulation  
**Limitations / cautions:** Never practice destructive commands on important data.

#### REAL-07 — Atlassian multi-site deletion — Apr 2022

**Link:** https://www.atlassian.com/blog/how-we-build/post-incident-review-april-2022-outage  
**Description:** A maintenance script deleted customer sites rather than intended app-level data. Explains authorization boundaries, recovery complexity and crisis communication.  
**Topics:** Cloud, Database, IncidentResponse  
**Learning format:** Original post-incident review  
**Access / environment:** Free reading; Web / hypothetical test app  
**Can I recreate a failure?** Conceptual / mock reproduction  
**Limitations / cautions:** Full SaaS recovery scenario is too large for a small lab.

#### REAL-08 — Meta/Facebook global backbone outage — Oct 4, 2021

**Link:** https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/  
**Description:** A maintenance command disconnected backbone connectivity, with effects on reachability and internal operational tools. Good for control-plane dependency and recovery discussions.  
**Topics:** Networking, DistributedSystems, IncidentResponse  
**Learning format:** Original technical report  
**Access / environment:** Free reading; Web / optional network emulator  
**Can I recreate a failure?** Conceptual / advanced  
**Limitations / cautions:** Production-scale backbone outage cannot be safely replicated at home.

#### REAL-09 — Slack network saturation outage — Jan 4, 2021

**Link:** https://slack.engineering/slacks-outage-on-january-4th-2021/  
**Description:** Traffic and network congestion cascaded into provisioning and web-tier problems; observability tooling itself was affected.  
**Topics:** Networking, Observability, Cloud  
**Learning format:** Original technical postmortem  
**Access / environment:** Free reading; Web / load test lab  
**Can I recreate a failure?** Partial local simulation  
**Limitations / cautions:** Use bounded test traffic, not stress testing against external services.

#### REAL-10 — Fastly global outage — Jun 8, 2021

**Link:** https://fastly.quic.co/blog/summary-of-june-8-outage  
**Description:** A latent software bug triggered by a valid customer configuration caused broad CDN errors. Useful for configuration validation, rollback and fault containment.  
**Topics:** Networking, Backend, Cloud  
**Learning format:** Original company incident summary  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Conceptual / partial  
**Limitations / cautions:** Source uses a Fastly-associated archival domain; follow company link carefully.

#### REAL-11 — Google Cloud multi-service outage — Jun 12, 2025

**Link:** https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW  
**Description:** Google Cloud public incident page showing cross-service API errors and a timeline of detection, mitigation and recovery.  
**Topics:** GCP, Cloud, DistributedSystems  
**Learning format:** Official status / incident report  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Conceptual / mock  
**Limitations / cautions:** Check the page for any newer comprehensive postmortem; mini reports can be preliminary.

### BLOG — Engineering blogs and official incident feeds

Ongoing publishers useful for finding more technical incidents, investigations and architecture lessons.

#### BLOG-01 — Cloudflare Blog — Outage / Deep Dives

**Link:** https://blog.cloudflare.com/  
**Description:** Detailed first-party writing on production networks, edge systems, security rules, data systems and failure analysis.  
**Topics:** Networking, Backend, Postmortem  
**Learning format:** Engineering blog  
**Access / environment:** Free; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Browse/search for outage and postmortem; not every article is an incident.

#### BLOG-02 — GitHub Engineering Blog

**Link:** https://github.blog/engineering/  
**Description:** Reports and engineering deep dives involving data storage, availability, large-scale web services and developer systems.  
**Topics:** Database, Cloud, DistributedSystems  
**Learning format:** Engineering blog  
**Access / environment:** Free; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Some outage writeups live under News & Insights rather than Engineering.

#### BLOG-03 — GitLab Engineering / Blog

**Link:** https://about.gitlab.com/blog/  
**Description:** Operational reports on CI/CD, GitLab.com, data systems, availability and reliability practices.  
**Topics:** Database, DevOps, IncidentResponse  
**Learning format:** Engineering blog  
**Access / environment:** Free; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Filter out product-release and marketing articles.

#### BLOG-04 — Slack Engineering

**Link:** https://slack.engineering/  
**Description:** Technical articles on backend operations, distributed systems and production failures.  
**Topics:** Backend, DistributedSystems, Observability  
**Learning format:** Engineering blog  
**Access / environment:** Free; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Many articles are design write-ups rather than incident reports.

#### BLOG-05 — Meta Engineering

**Link:** https://engineering.fb.com/  
**Description:** First-party analyses of networking, large-scale systems and operational incidents.  
**Topics:** Networking, DistributedSystems  
**Learning format:** Engineering blog  
**Access / environment:** Free; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Select deep technical write-ups for failure study.

#### BLOG-06 — Netflix TechBlog

**Link:** https://netflixtechblog.com/  
**Description:** Well-known source for reliability architecture, chaos engineering and operating large distributed applications.  
**Topics:** Chaos, DistributedSystems, Backend  
**Learning format:** Engineering blog  
**Access / environment:** Free to read; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Not a structured troubleshooting curriculum.

#### BLOG-07 — AWS Builders Library

**Link:** https://builder.aws.com/learn/topics/builders-library  
**Description:** Amazon engineering articles on operating dependable services, retries, deployments, overload and failure containment.  
**Topics:** AWS, DistributedSystems, SRE  
**Learning format:** Engineering practices  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Conceptual / self-designed  
**Limitations / cautions:** These are engineering essays, not a collection of postmortems.

#### BLOG-08 — Google Cloud Service Health

**Link:** https://status.cloud.google.com/  
**Description:** Official status incident history across GCP services; use to identify specific real outages for follow-up analysis.  
**Topics:** GCP, Cloud, Postmortem  
**Learning format:** Incident feed  
**Access / environment:** Free; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Duplicate of cloud incident directory by purpose; listed here as a feed, not a new platform.

### GIT — Open-source GitHub repositories and exercise collections

Reusable exercises, experiment code, datasets and reference examples.

#### GIT-01 — bregman-arie/devops-exercises

**Link:** https://github.com/bregman-arie/devops-exercises  
**Description:** A broad open-source set of DevOps/SRE exercises and questions covering Linux, Docker, AWS, Kubernetes, Terraform, DNS, databases, and automation.  
**Topics:** Linux, AWS, Docker, Kubernetes, Terraform, Database  
**Learning format:** Exercises / questions  
**Access / environment:** Free source; GitHub + local environment  
**Can I recreate a failure?** Varies  
**Limitations / cautions:** Most entries are questions and prompts, not provisioned broken systems.

#### GIT-02 — chaos-mesh/chaos-mesh

**Link:** https://github.com/chaos-mesh/chaos-mesh  
**Description:** Source and implementation reference for the Kubernetes Chaos Mesh operators, experiments and dashboard.  
**Topics:** Kubernetes, Chaos  
**Learning format:** Tool source  
**Access / environment:** Open source; GitHub / own cluster  
**Can I recreate a failure?** Yes with setup  
**Limitations / cautions:** Use project docs to match release and supported environment.

#### GIT-03 — litmuschaos/litmus

**Link:** https://github.com/litmuschaos/litmus  
**Description:** Source, examples and documentation links for a Kubernetes-oriented chaos engineering framework.  
**Topics:** Kubernetes, Chaos  
**Learning format:** Tool source  
**Access / environment:** Open source; GitHub / own cluster  
**Can I recreate a failure?** Yes with setup  
**Limitations / cautions:** Requires orchestrated infrastructure, permissions and teardown.

#### GIT-04 — microsoft/chaos-studio

**Link:** https://github.com/microsoft/chaos-studio  
**Description:** Community companion repository with samples and scenarios for Azure Chaos Studio.  
**Topics:** Azure, Chaos, Cloud  
**Learning format:** Samples / scenario code  
**Access / environment:** Free source; Azure costs possible; GitHub + Azure  
**Can I recreate a failure?** Yes with setup  
**Limitations / cautions:** Repository is companion tooling, not the managed service source.

#### GIT-05 — danluu/post-mortems

**Link:** https://github.com/danluu/post-mortems  
**Description:** The raw open-source curated link list supporting investigations across decades of company failures.  
**Topics:** Postmortem, MultiTopic  
**Learning format:** Dataset / link index  
**Access / environment:** Free; GitHub  
**Can I recreate a failure?** No  
**Limitations / cautions:** Duplicates COL-03 by design; source-code access for the same index.

#### GIT-06 — icco/postmortems

**Link:** https://github.com/icco/postmortems  
**Description:** Open, annotated incident corpus and source behind postmortems.app with machine-readable metadata and categories.  
**Topics:** Postmortem, MultiTopic  
**Learning format:** Dataset / structured metadata  
**Access / environment:** Free; GitHub / JSON  
**Can I recreate a failure?** No  
**Limitations / cautions:** Duplicate underlying resource for COL-02; useful for automated personal indexing.

#### GIT-07 — AWS FIS Workshop Source

**Link:** https://github.com/aws-samples/aws-fault-injection-simulator-workshop  
**Description:** Source code and setup assets for an AWS fault injection workshop that can be inspected before attempting cloud experiments.  
**Topics:** AWS, Chaos  
**Learning format:** Workshop repository  
**Access / environment:** Free source; AWS costs possible; GitHub + AWS  
**Can I recreate a failure?** Yes with setup  
**Limitations / cautions:** Also indexed as CLOUD-03 to make it discoverable under both topic and source type.

### REF — SRE method, troubleshooting and postmortem references

Authoritative material on how to investigate, manage, and learn from failures; not inherently interactive.

#### REF-01 — Google SRE Book

**Link:** https://sre.google/sre-book/table-of-contents/  
**Description:** Foundational site reliability text. Relevant chapters include monitoring, effective troubleshooting, incident management, testing reliability, overload and cascading failures.  
**Topics:** SRE, DistributedSystems, Observability  
**Learning format:** Reference book  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Conceptual examples  
**Limitations / cautions:** A book, not a failure platform. Select individual chapters as needed later.

#### REF-02 — Google SRE Workbook — Postmortem Culture

**Link:** https://sre.google/workbook/postmortem-culture/  
**Description:** How to produce useful blameless incident reports, identify system contributors and convert lessons into measurable prevention tasks.  
**Topics:** SRE, IncidentResponse, Postmortem  
**Learning format:** Methods / templates  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Distinguishes technical remediation from superficial blame.

#### REF-03 — Google SRE — Incident Management Guide

**Link:** https://sre.google/resources/practices-and-processes/incident-management-guide/  
**Description:** A concise operational framework for coordinating an incident, mitigation, communication and learning afterward.  
**Topics:** SRE, IncidentResponse  
**Learning format:** Runbook / method  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** No  
**Limitations / cautions:** Organizational process guide rather than a simulator.

#### REF-04 — Google SRE — Addressing Cascading Failures

**Link:** https://sre.google/sre-book/addressing-cascading-failures/  
**Description:** Detailed explanation of overload feedback loops, retry storms, unhealthy backends and failure propagation. Useful for reasoning about multi-service incidents.  
**Topics:** DistributedSystems, Backend, Observability  
**Learning format:** Engineering reference  
**Access / environment:** Free reading; Web  
**Can I recreate a failure?** Can design toy experiments  
**Limitations / cautions:** The architecture examples are conceptual rather than ready-to-run labs.

#### REF-05 — AWS FIS Stop Conditions

**Link:** https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html  
**Description:** AWS official stop-condition design using CloudWatch alarms; the guardrail to consult before any AWS fault-injection exercise.  
**Topics:** AWS, Chaos, Safety  
**Learning format:** Safety reference  
**Access / environment:** Free docs; Web  
**Can I recreate a failure?** Supporting material  
**Limitations / cautions:** Same as CLOUD-02, repeated as a critical reference.

#### REF-06 — Terraform State Recovery Guide

**Link:** https://docs.hashicorp.com/terraform/cli/state/recover  
**Description:** Official operations reference for recovering Terraform backend state, not just recreating resource definitions.  
**Topics:** Terraform, Cloud  
**Learning format:** Recovery reference  
**Access / environment:** Free docs; Web  
**Can I recreate a failure?** Can rehearse in sandbox  
**Limitations / cautions:** Same resource as CLOUD-06, cross-listed for discoverability.

## 4. Browse by technical domain

This index is **multi-label**: one resource can appear under multiple technologies. IDs resolve to complete entries in Section 3. It is not a prerequisite graph.

### AWS / EC2 / IAM / cloud operations

**LAB-07** ([AWS GameDay](https://aws.amazon.com/gameday/)), **CLOUD-01** ([AWS Fault Injection Service (FIS)](https://aws.amazon.com/fis/)), **CLOUD-02** ([AWS FIS Stop Conditions](https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html)), **CLOUD-03** ([AWS FIS Workshop Repository](https://github.com/aws-samples/aws-fault-injection-simulator-workshop)), **CLOUD-04** ([AWS Workshop Studio Catalog](https://catalog.workshops.aws/)), **CLOUD-05** ([Terraform — Manage State](https://developer.hashicorp.com/terraform/tutorials/state)), **CLOUD-06** ([Terraform — Manage Resource Drift](https://docs.hashicorp.com/terraform/tutorials/state/resource-drift)), **COL-07** ([AWS Post-Event Summaries](https://aws.amazon.com/premiumsupport/technology/pes/)), **REAL-01** ([AWS us-east-1 disruption — Dec 7, 2021](https://aws.amazon.com/message/12721//)), **BLOG-07** ([AWS Builders Library](https://builder.aws.com/learn/topics/builders-library)), **GIT-01** ([bregman-arie/devops-exercises](https://github.com/bregman-arie/devops-exercises)), **GIT-07** ([AWS FIS Workshop Source](https://github.com/aws-samples/aws-fault-injection-simulator-workshop)), **REF-05** ([AWS FIS Stop Conditions](https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html)). Additional relevant resources: **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)).

### Terraform / infrastructure-as-code

**LAB-01** ([SadServers](https://sadservers.com/scenarios)), **CLOUD-05** ([Terraform — Manage State](https://developer.hashicorp.com/terraform/tutorials/state)), **CLOUD-06** ([Terraform — Manage Resource Drift](https://docs.hashicorp.com/terraform/tutorials/state/resource-drift)), **CLOUD-07** ([Terraform — Recover State](https://docs.hashicorp.com/terraform/cli/state/recover)), **GIT-01** ([bregman-arie/devops-exercises](https://github.com/bregman-arie/devops-exercises)), **REF-06** ([Terraform State Recovery Guide](https://docs.hashicorp.com/terraform/cli/state/recover)). Additional relevant resources: **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)), **LAB-11** ([KodeKloud Free Labs](https://kodekloud.com/studio/labs)).

### Linux / system services

**LAB-01** ([SadServers](https://sadservers.com/scenarios)), **LAB-02** ([Learn by Fixing](https://www.learnbyfixing.com/scenarios/)), **LAB-03** ([Killercoda](https://killercoda.com/about)), **LAB-06** ([Killer Shell Simulators](https://killer.sh/)), **REAL-06** ([GitLab database deletion — Jan 31, 2017](https://about.gitlab.com/blog/postmortem-of-database-outage-of-january-31/)), **GIT-01** ([bregman-arie/devops-exercises](https://github.com/bregman-arie/devops-exercises)). Additional relevant resources: **LAB-08** ([KodeKloud Engineer](https://engineer.kodekloud.com/)), **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)), **LAB-11** ([KodeKloud Free Labs](https://kodekloud.com/studio/labs)).

### Docker and containers

**LAB-01** ([SadServers](https://sadservers.com/scenarios)), **LAB-02** ([Learn by Fixing](https://www.learnbyfixing.com/scenarios/)), **LAB-03** ([Killercoda](https://killercoda.com/about)), **GIT-01** ([bregman-arie/devops-exercises](https://github.com/bregman-arie/devops-exercises)). Additional relevant resources: **LAB-08** ([KodeKloud Engineer](https://engineer.kodekloud.com/)), **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)).

### Kubernetes / cluster operations

**LAB-01** ([SadServers](https://sadservers.com/scenarios)), **LAB-02** ([Learn by Fixing](https://www.learnbyfixing.com/scenarios/)), **LAB-03** ([Killercoda](https://killercoda.com/about)), **LAB-04** ([Killercoda CKA Scenarios](https://killercoda.com/cka)), **LAB-05** ([Killercoda CKS Scenarios](https://killercoda.com/cks)), **LAB-06** ([Killer Shell Simulators](https://killer.sh/)), **CHAOS-01** ([Chaos Mesh](https://chaos-mesh.org/docs/)), **CHAOS-02** ([LitmusChaos](https://docs.litmuschaos.io/)), **CHAOS-03** ([Litmus ChaosHub](https://hub.litmuschaos.io/)), **COL-04** ([Kubernetes Failure Stories](https://k8s.af/)), **GIT-01** ([bregman-arie/devops-exercises](https://github.com/bregman-arie/devops-exercises)), **GIT-02** ([chaos-mesh/chaos-mesh](https://github.com/chaos-mesh/chaos-mesh)), **GIT-03** ([litmuschaos/litmus](https://github.com/litmuschaos/litmus)). Additional relevant resources: **LAB-08** ([KodeKloud Engineer](https://engineer.kodekloud.com/)), **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)), **LAB-10** ([KodeKloud Kubernetes Troubleshooting](https://kodekloud.com/courses/kubernetes-troubleshooting-for-application-developers)), **LAB-11** ([KodeKloud Free Labs](https://kodekloud.com/studio/labs)).

### DNS / routing / network

**LAB-03** ([Killercoda](https://killercoda.com/about)), **LAB-04** ([Killercoda CKA Scenarios](https://killercoda.com/cka)), **LAB-05** ([Killercoda CKS Scenarios](https://killercoda.com/cks)), **CLOUD-08** ([Azure Chaos Studio](https://learn.microsoft.com/en-us/azure/chaos-studio/chaos-studio-overview)), **CLOUD-09** ([Azure Chaos Studio Scenario Catalog](https://learn.microsoft.com/azure/chaos-studio/chaos-studio-scenarios)), **CHAOS-01** ([Chaos Mesh](https://chaos-mesh.org/docs/)), **COL-04** ([Kubernetes Failure Stories](https://k8s.af/)), **COL-06** ([Cloudflare Status](https://www.cloudflarestatus.com/)), **REAL-01** ([AWS us-east-1 disruption — Dec 7, 2021](https://aws.amazon.com/message/12721//)), **REAL-02** ([Cloudflare WAF regex/CPU outage — Jul 2, 2019](https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/)), **REAL-03** ([Cloudflare BGP configuration outage — Jun 21, 2022](https://blog.cloudflare.com/cloudflare-outage-on-june-21-2022/)), **REAL-04** ([Cloudflare Bot Management data outage — Nov 18, 2025](https://blog.cloudflare.com/18-november-2025-outage/)), **REAL-05** ([GitHub network partition / MySQL failover — Oct 2018](https://github.blog/news-insights/company-news/oct21-post-incident-analysis/)), **REAL-08** ([Meta/Facebook global backbone outage — Oct 4, 2021](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/)), **REAL-09** ([Slack network saturation outage — Jan 4, 2021](https://slack.engineering/slacks-outage-on-january-4th-2021/)), **REAL-10** ([Fastly global outage — Jun 8, 2021](https://fastly.quic.co/blog/summary-of-june-8-outage)), **BLOG-01** ([Cloudflare Blog — Outage / Deep Dives](https://blog.cloudflare.com/)), **BLOG-05** ([Meta Engineering](https://engineering.fb.com/)). Additional relevant resources: **LAB-10** ([KodeKloud Kubernetes Troubleshooting](https://kodekloud.com/courses/kubernetes-troubleshooting-for-application-developers)).

### SQL / NoSQL / data recovery

**LAB-01** ([SadServers](https://sadservers.com/scenarios)), **LAB-02** ([Learn by Fixing](https://www.learnbyfixing.com/scenarios/)), **CLOUD-09** ([Azure Chaos Studio Scenario Catalog](https://learn.microsoft.com/azure/chaos-studio/chaos-studio-scenarios)), **REAL-04** ([Cloudflare Bot Management data outage — Nov 18, 2025](https://blog.cloudflare.com/18-november-2025-outage/)), **REAL-05** ([GitHub network partition / MySQL failover — Oct 2018](https://github.blog/news-insights/company-news/oct21-post-incident-analysis/)), **REAL-06** ([GitLab database deletion — Jan 31, 2017](https://about.gitlab.com/blog/postmortem-of-database-outage-of-january-31/)), **REAL-07** ([Atlassian multi-site deletion — Apr 2022](https://www.atlassian.com/blog/how-we-build/post-incident-review-april-2022-outage)), **BLOG-02** ([GitHub Engineering Blog](https://github.blog/engineering/)), **BLOG-03** ([GitLab Engineering / Blog](https://about.gitlab.com/blog/)), **GIT-01** ([bregman-arie/devops-exercises](https://github.com/bregman-arie/devops-exercises)). Additional relevant resources: **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)).

### Backend / API / service failures

**REAL-02** ([Cloudflare WAF regex/CPU outage — Jul 2, 2019](https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/)), **REAL-04** ([Cloudflare Bot Management data outage — Nov 18, 2025](https://blog.cloudflare.com/18-november-2025-outage/)), **REAL-10** ([Fastly global outage — Jun 8, 2021](https://fastly.quic.co/blog/summary-of-june-8-outage)), **BLOG-01** ([Cloudflare Blog — Outage / Deep Dives](https://blog.cloudflare.com/)), **BLOG-04** ([Slack Engineering](https://slack.engineering/)), **BLOG-06** ([Netflix TechBlog](https://netflixtechblog.com/)), **REF-04** ([Google SRE — Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)).

### Distributed systems / consistency / cascades

**LAB-07** ([AWS GameDay](https://aws.amazon.com/gameday/)), **CHAOS-01** ([Chaos Mesh](https://chaos-mesh.org/docs/)), **CHAOS-04** ([Chaos Toolkit](https://github.com/chaostoolkit/chaostoolkit)), **CHAOS-05** ([Netflix Chaos Monkey](https://github.com/Netflix/chaosmonkey)), **REAL-01** ([AWS us-east-1 disruption — Dec 7, 2021](https://aws.amazon.com/message/12721//)), **REAL-05** ([GitHub network partition / MySQL failover — Oct 2018](https://github.blog/news-insights/company-news/oct21-post-incident-analysis/)), **REAL-08** ([Meta/Facebook global backbone outage — Oct 4, 2021](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/)), **REAL-11** ([Google Cloud multi-service outage — Jun 12, 2025](https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW)), **BLOG-02** ([GitHub Engineering Blog](https://github.blog/engineering/)), **BLOG-04** ([Slack Engineering](https://slack.engineering/)), **BLOG-05** ([Meta Engineering](https://engineering.fb.com/)), **BLOG-06** ([Netflix TechBlog](https://netflixtechblog.com/)), **BLOG-07** ([AWS Builders Library](https://builder.aws.com/learn/topics/builders-library)), **REF-01** ([Google SRE Book](https://sre.google/sre-book/table-of-contents/)), **REF-04** ([Google SRE — Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)).

### Monitoring / logging / metrics

**LAB-01** ([SadServers](https://sadservers.com/scenarios)), **CLOUD-01** ([AWS Fault Injection Service (FIS)](https://aws.amazon.com/fis/)), **CLOUD-02** ([AWS FIS Stop Conditions](https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html)), **CLOUD-04** ([AWS Workshop Studio Catalog](https://catalog.workshops.aws/)), **CHAOS-02** ([LitmusChaos](https://docs.litmuschaos.io/)), **REAL-02** ([Cloudflare WAF regex/CPU outage — Jul 2, 2019](https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/)), **REAL-09** ([Slack network saturation outage — Jan 4, 2021](https://slack.engineering/slacks-outage-on-january-4th-2021/)), **BLOG-04** ([Slack Engineering](https://slack.engineering/)), **REF-01** ([Google SRE Book](https://sre.google/sre-book/table-of-contents/)), **REF-04** ([Google SRE — Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)). Additional relevant resources: **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)).

### Chaos engineering / fault injection

**CLOUD-01** ([AWS Fault Injection Service (FIS)](https://aws.amazon.com/fis/)), **CLOUD-02** ([AWS FIS Stop Conditions](https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html)), **CLOUD-03** ([AWS FIS Workshop Repository](https://github.com/aws-samples/aws-fault-injection-simulator-workshop)), **CLOUD-04** ([AWS Workshop Studio Catalog](https://catalog.workshops.aws/)), **CLOUD-08** ([Azure Chaos Studio](https://learn.microsoft.com/en-us/azure/chaos-studio/chaos-studio-overview)), **CLOUD-09** ([Azure Chaos Studio Scenario Catalog](https://learn.microsoft.com/azure/chaos-studio/chaos-studio-scenarios)), **CHAOS-01** ([Chaos Mesh](https://chaos-mesh.org/docs/)), **CHAOS-02** ([LitmusChaos](https://docs.litmuschaos.io/)), **CHAOS-03** ([Litmus ChaosHub](https://hub.litmuschaos.io/)), **CHAOS-04** ([Chaos Toolkit](https://github.com/chaostoolkit/chaostoolkit)), **CHAOS-05** ([Netflix Chaos Monkey](https://github.com/Netflix/chaosmonkey)), **CHAOS-06** ([Netflix Simian Army (historical)](https://github.com/Netflix/SimianArmy)), **CHAOS-07** ([Gremlin](https://www.gremlin.com/product)), **BLOG-06** ([Netflix TechBlog](https://netflixtechblog.com/)), **GIT-02** ([chaos-mesh/chaos-mesh](https://github.com/chaos-mesh/chaos-mesh)), **GIT-03** ([litmuschaos/litmus](https://github.com/litmuschaos/litmus)), **GIT-04** ([microsoft/chaos-studio](https://github.com/microsoft/chaos-studio)), **GIT-07** ([AWS FIS Workshop Source](https://github.com/aws-samples/aws-fault-injection-simulator-workshop)), **REF-05** ([AWS FIS Stop Conditions](https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html)).

### Incident response / operations

**LAB-07** ([AWS GameDay](https://aws.amazon.com/gameday/)), **CLOUD-10** ([Google Cloud Incident Reports](https://status.cloud.google.com/)), **COL-05** ([GitHub Status / History](https://www.githubstatus.com/)), **COL-06** ([Cloudflare Status](https://www.cloudflarestatus.com/)), **REAL-03** ([Cloudflare BGP configuration outage — Jun 21, 2022](https://blog.cloudflare.com/cloudflare-outage-on-june-21-2022/)), **REAL-06** ([GitLab database deletion — Jan 31, 2017](https://about.gitlab.com/blog/postmortem-of-database-outage-of-january-31/)), **REAL-07** ([Atlassian multi-site deletion — Apr 2022](https://www.atlassian.com/blog/how-we-build/post-incident-review-april-2022-outage)), **REAL-08** ([Meta/Facebook global backbone outage — Oct 4, 2021](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/)), **BLOG-03** ([GitLab Engineering / Blog](https://about.gitlab.com/blog/)), **REF-02** ([Google SRE Workbook — Postmortem Culture](https://sre.google/workbook/postmortem-culture/)), **REF-03** ([Google SRE — Incident Management Guide](https://sre.google/resources/practices-and-processes/incident-management-guide/)). Additional relevant resources: **LAB-08** ([KodeKloud Engineer](https://engineer.kodekloud.com/)), **LAB-10** ([KodeKloud Kubernetes Troubleshooting](https://kodekloud.com/courses/kubernetes-troubleshooting-for-application-developers)).

### Postmortems / incident reviews

**COL-01** ([postmortem.io](https://postmortem.io/)), **COL-02** ([postmortems.app](https://postmortems.app/)), **COL-03** ([danluu/post-mortems](https://github.com/danluu/post-mortems)), **COL-04** ([Kubernetes Failure Stories](https://k8s.af/)), **COL-07** ([AWS Post-Event Summaries](https://aws.amazon.com/premiumsupport/technology/pes/)), **BLOG-01** ([Cloudflare Blog — Outage / Deep Dives](https://blog.cloudflare.com/)), **BLOG-08** ([Google Cloud Service Health](https://status.cloud.google.com/)), **GIT-05** ([danluu/post-mortems](https://github.com/danluu/post-mortems)), **GIT-06** ([icco/postmortems](https://github.com/icco/postmortems)), **REF-02** ([Google SRE Workbook — Postmortem Culture](https://sre.google/workbook/postmortem-culture/)).

### Multi-cloud architecture

**LAB-07** ([AWS GameDay](https://aws.amazon.com/gameday/)), **CLOUD-01** ([AWS Fault Injection Service (FIS)](https://aws.amazon.com/fis/)), **CLOUD-03** ([AWS FIS Workshop Repository](https://github.com/aws-samples/aws-fault-injection-simulator-workshop)), **CLOUD-04** ([AWS Workshop Studio Catalog](https://catalog.workshops.aws/)), **CLOUD-05** ([Terraform — Manage State](https://developer.hashicorp.com/terraform/tutorials/state)), **CLOUD-06** ([Terraform — Manage Resource Drift](https://docs.hashicorp.com/terraform/tutorials/state/resource-drift)), **CLOUD-07** ([Terraform — Recover State](https://docs.hashicorp.com/terraform/cli/state/recover)), **CLOUD-08** ([Azure Chaos Studio](https://learn.microsoft.com/en-us/azure/chaos-studio/chaos-studio-overview)), **CLOUD-10** ([Google Cloud Incident Reports](https://status.cloud.google.com/)), **CHAOS-04** ([Chaos Toolkit](https://github.com/chaostoolkit/chaostoolkit)), **CHAOS-05** ([Netflix Chaos Monkey](https://github.com/Netflix/chaosmonkey)), **CHAOS-06** ([Netflix Simian Army (historical)](https://github.com/Netflix/SimianArmy)), **CHAOS-07** ([Gremlin](https://www.gremlin.com/product)), **COL-05** ([GitHub Status / History](https://www.githubstatus.com/)), **COL-07** ([AWS Post-Event Summaries](https://aws.amazon.com/premiumsupport/technology/pes/)), **REAL-03** ([Cloudflare BGP configuration outage — Jun 21, 2022](https://blog.cloudflare.com/cloudflare-outage-on-june-21-2022/)), **REAL-07** ([Atlassian multi-site deletion — Apr 2022](https://www.atlassian.com/blog/how-we-build/post-incident-review-april-2022-outage)), **REAL-09** ([Slack network saturation outage — Jan 4, 2021](https://slack.engineering/slacks-outage-on-january-4th-2021/)), **REAL-10** ([Fastly global outage — Jun 8, 2021](https://fastly.quic.co/blog/summary-of-june-8-outage)), **REAL-11** ([Google Cloud multi-service outage — Jun 12, 2025](https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW)), **BLOG-02** ([GitHub Engineering Blog](https://github.blog/engineering/)), **BLOG-08** ([Google Cloud Service Health](https://status.cloud.google.com/)), **GIT-04** ([microsoft/chaos-studio](https://github.com/microsoft/chaos-studio)), **REF-06** ([Terraform State Recovery Guide](https://docs.hashicorp.com/terraform/cli/state/recover)). Additional relevant resources: **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)).

### Azure

**CLOUD-08** ([Azure Chaos Studio](https://learn.microsoft.com/en-us/azure/chaos-studio/chaos-studio-overview)), **CLOUD-09** ([Azure Chaos Studio Scenario Catalog](https://learn.microsoft.com/azure/chaos-studio/chaos-studio-scenarios)), **GIT-04** ([microsoft/chaos-studio](https://github.com/microsoft/chaos-studio)). Additional relevant resources: **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)).

### Google Cloud

**CLOUD-10** ([Google Cloud Incident Reports](https://status.cloud.google.com/)), **REAL-11** ([Google Cloud multi-service outage — Jun 12, 2025](https://status.cloud.google.com/incidents/ow5i3PPK96RduMcb1SsW)), **BLOG-08** ([Google Cloud Service Health](https://status.cloud.google.com/)). Additional relevant resources: **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)).

### Security / access / policy

**LAB-03** ([Killercoda](https://killercoda.com/about)), **LAB-05** ([Killercoda CKS Scenarios](https://killercoda.com/cks)), **LAB-06** ([Killer Shell Simulators](https://killer.sh/)). Additional relevant resources: **LAB-10** ([KodeKloud Kubernetes Troubleshooting](https://kodekloud.com/courses/kubernetes-troubleshooting-for-application-developers)).

### SRE methods

**CHAOS-07** ([Gremlin](https://www.gremlin.com/product)), **BLOG-07** ([AWS Builders Library](https://builder.aws.com/learn/topics/builders-library)), **REF-01** ([Google SRE Book](https://sre.google/sre-book/table-of-contents/)), **REF-02** ([Google SRE Workbook — Postmortem Culture](https://sre.google/workbook/postmortem-culture/)), **REF-03** ([Google SRE — Incident Management Guide](https://sre.google/resources/practices-and-processes/incident-management-guide/)).

### DevOps tooling

**BLOG-03** ([GitLab Engineering / Blog](https://about.gitlab.com/blog/)). Additional relevant resources: **LAB-08** ([KodeKloud Engineer](https://engineer.kodekloud.com/)), **LAB-09** ([KodeKloud Playgrounds](https://kodekloud.com/playgrounds)), **LAB-11** ([KodeKloud Free Labs](https://kodekloud.com/studio/labs)).

### Experiment safety

**REF-05** ([AWS FIS Stop Conditions](https://docs.aws.amazon.com/fis/latest/userguide/stop-conditions.html)).

### Storage

**LAB-04** ([Killercoda CKA Scenarios](https://killercoda.com/cka)). Additional relevant resources: **LAB-10** ([KodeKloud Kubernetes Troubleshooting](https://kodekloud.com/courses/kubernetes-troubleshooting-for-application-developers)).

## 5. Browse by failure mode

These are **search starting points**, not promised ready-made exercises. The exact case and reproduction conditions must be checked on the linked resource.

| Failure pattern | Resources to inspect | Investigation themes |
|---|---|---|
| DNS resolution fails | LAB-01, LAB-02, LAB-03, COL-04, CLOUD-09, LAB-10 | DNS resolver, service discovery, cached results, network policies |
| Container crashes or cannot start | LAB-01, LAB-02, LAB-03, GIT-01, LAB-08, LAB-10 | Image, entrypoint, environment, mount, port, OOM, permissions |
| Kubernetes pod Pending / service unreachable | LAB-02, LAB-03, LAB-04, COL-04, LAB-08, LAB-10 | Scheduler events, affinity, quota, selectors, endpoints, CNI |
| EC2 resurfaces after termination / desired capacity | LAB-07, CLOUD-01, COL-01, GIT-01, LAB-09 | Spot request or fleet ownership, controllers, CloudTrail, AWS resource lifecycle |
| Terraform state missing or drift detected | CLOUD-05, CLOUD-06, CLOUD-07, LAB-09 | Local versus remote state, import, refresh-only, actual versus desired state |
| Database unavailable / replication inconsistent | LAB-01, REAL-05, REAL-06, COL-01, LAB-09 | Replica leadership, failover, lag, backup integrity, recovery |
| Misconfiguration creates global outage | REAL-02, REAL-03, REAL-04, REAL-07, REAL-10 | Change boundaries, validation, canaries, rollback |
| Network partition / routing failure | REAL-03, REAL-05, REAL-08, REAL-09 | BGP, VPC/TGW, quorum, split brain, connectivity |
| CPU pressure and cascading overload | REAL-02, REAL-09, REF-04, CHAOS-01 | Resource exhaustion, timeouts, retry amplification, backoff |
| Accidental deletion / data loss | REAL-06, REAL-07, CLOUD-07 | Backups, restores, soft delete, IAM, destructive operation guardrails |
| Telemetry unavailable during incident | REAL-09, REF-03, LAB-01 | Out-of-band visibility, dependency isolation, incident coordination |
| Intentional resilience / failover testing | CLOUD-01, CLOUD-07, CHAOS-01, CHAOS-02, CHAOS-04 | Steady state, bounded blast radius, abort thresholds, recovery assertions |

## 6. How to study reports and build safe reproductions

This is a **study template**, not a curriculum or ordered list of what to learn.

### 6.1 Incident evidence template

```yaml
resource_id: REAL-XX
incident_name:
source_url:
company_or_lab:
date_of_original_incident:
symptoms_visible_to_users:
initial_hypotheses: []
evidence: []  # logs, traces, metrics, audit events, configuration diffs
trigger:
contributing_factors: []
blast_radius:
mitigation:
permanent_correction:
verification_evidence:
preventative_controls: []
safe_reproduction_possible: unknown
notes:
```

### 6.2 Safe reproduction checklist

- [ ] Work in a disposable lab, not a production or shared environment.
- [ ] Write down the known **healthy baseline** and measurable success criteria.
- [ ] Define the single failure to introduce and estimate the **blast radius**.
- [ ] Confirm permissions, isolation and a way to stop the experiment.
- [ ] For cloud tests, enable budgets/alerts and understand metered resources; a budget is **not** an automatic hard cap.
- [ ] Take snapshots/backups when state is important; verify how restoration works.
- [ ] Introduce the fault, collect evidence, propose a hypothesis, apply one correction.
- [ ] Verify behavior independently and check it remains healthy after any controller reconciles.
- [ ] Delete **controllers, persistent requests, fleets, schedules and resource dependencies** before terminating child resources.
- [ ] Recheck instances, persistent disks/EBS, public IPs, load balancers, NAT gateways and other chargeable resources.
- [ ] Record an incident review with contributing factors, not only the final command.

### 6.3 Caution from the EC2 Spot replacement incident

The incident that motivated this catalog showed that terminating EC2 instances is not equivalent to canceling the underlying Spot capacity request or fleet. A launch audit can reveal the AWS service-linked role, while separate configuration evidence is needed to find the controller to disable. This is an ideal **resource lifecycle / reconciliation** case study, but do not deliberately recreate it using costly instances without a verified cleanup plan.

## 7. Verification, pricing and limitations

- All primary URLs in this edition were visited or supported by web search results available on **24 October 2026**. This is not a guarantee that every sub-link on those sites works or that plans stay unchanged.
- Original reports are preferred for incident facts. The aggregators in COL and GIT are discovery tools, and sometimes their summaries simplify or omit contributing causes.
- The **same resource may be cross-listed** under several categories (for example, a tool’s product documentation versus its GitHub source). That is intentional, not an additional learning platform.
- Historical posts remain relevant even when software versions have changed. Do not reproduce legacy commands uncritically on modern kernels, Kubernetes or cloud services.
- The `REPLAY` field is a planning estimate. A postmortem may be suitable for a *simplified toy reproduction*, not faithful recreation of a company-scale outage.
- Costs vary. Free reading does not imply free infrastructure, free experimentation, unlimited sessions, or a guarantee of credits.
- Some catalog resources are task/assessment platforms; they are listed for their operational practice value, not represented as pure outage simulations.
- For security-lab content, use only systems you own or environments explicitly provided for testing.

## 8. Future roadmap placeholder

> **Intentionally deferred.** We will later assess the catalog against existing skills, hands-on availability, cost, reproducibility, technical depth and overlap. Only then decide **what to learn first, what to learn later and which incident to recreate**. No priority ranking is implied by this document.

Potential later artifacts (not part of this edition): comparison scorecard, chosen learning order, incident lab backlog, completed-investigation log and a recurring review of newly published company postmortems.

---

**Catalog coverage (unique listed records, including explicit cross-listings):** 63 entries across 8 top-level resource types.  
**Catalog maintenance note:** Keep IDs stable; when a URL becomes unavailable, add an archive/alternative in the existing record rather than renumbering the catalog.
