# MLOps Workflow Primer (Local Kubernetes w/ scikit-learn)

This repository is a **small, end‑to‑end MLOps / ML‑workflow primer** designed to run **entirely on a local machine** using **Vagrant + Kubernetes (containerd)**.

The goal is **not** model sophistication, but to demonstrate **core ML workflow fundamentals** as simple as possible:

- Data ingestion from an external source
- Batch training as a Kubernetes Job
- Persisted model artifacts
- Online inference via an API
- Separation of training and serving concerns
- Reproducible, infrastructure‑aware setup

---

## Project Rationale

This project aims to be a learning/simulation environment for DevOps / Platform / Site-Reliability Engineers wanting to transition to MLOps.

It exists to demonstrate **ML workflow fundamentals** in a way that is local, inspectable, reproducible, and infrastructure‑aware.

It is designed to be read, modified, and extended — not treated as a black box. That is why it uses purely open-source tools running locally - those that can be tinkered without worry in the safe confines of one's personal computer. As such, the following design considerations were intentionally made:

- **No public image registry**

  - Images are built locally and imported directly into containerd.
  - Reflects on‑prem / restricted environments

- **Simple model**

  - Focus is workflow, not on the model accuracy & sophistication
  - MLOps is a separate discipline to Data Scientists, Machine Learning Engineers so intend to highlight automation and workflows, and not the python source code

- **Explicit scripts & manifests**

  - Avoids magic abstractions
  - Easy to reason about for learning purposes

---

## Training workflow

### Data source

The training workflow uses **public financial time‑series data** fetched at runtime via the `yfinance` library.

- Data type: historical price data (e.g. open / close / volume)
- Source: Yahoo Finance (via yfinance)
- Scope: a single financial instrument over a recent time window

The data is retrieved **inside the training container at job execution time**, ensuring:

- no bundled datasets
- no manual data downloads
- a realistic external‑data dependency

---

### Training objective

The goal of the training job is intentionally simple:

> Learn a relationship from historical price data and produce a model capable of predicting **next‑day price direction** (binary classification: UP or DOWN).

This is **not** intended to be a production‑grade financial model.
The emphasis is on **workflow mechanics**, not predictive performance.

---

### Feature preparation

At a high level, the training job:

1. Downloads historical time‑series data
2. Extracts basic numerical features from the data

   - recent prices
   - simple derived values (e.g. rolling statistics)

3. Constructs a supervised learning dataset

   - features derived from past values
   - target representing a future value

All feature logic lives entirely inside the training container, keeping the job self‑contained.

---

### Model training

- Framework: **scikit‑learn**
- Model: a lightweight binary classification model (RandomForestClassifier)

The model is chosen to be:

- fast to train
- deterministic
- easy to serialize and reload

This keeps training times short and makes the workflow easy to inspect.

---

### Model artifact

After training completes:

- The trained model is serialized to disk
- The artifact is written to a shared location
- No model state is kept in memory after job completion

This mirrors a common production pattern:

> **Training produces artifacts; serving consumes artifacts**

---

### Why this workflow

This training pipeline is intentionally minimal, but demonstrates key ML workflow concepts:

- Batch training as a Kubernetes Job
- External data dependency at runtime
- Clear separation of training and inference
- Artifact‑based handoff between components

---

## Prerequisites

On the host machine:

- Docker
- Vagrant
- VirtualBox
- kubectl (Optional)

You will only ever need Vagrant, but vagrant needs docker and virtualbox as part of the workflow it is programmed in the Vagrantfile in setting up the development environment. You can ssh inside the VM and run kubectl from inside the nodes, but you may want to run kubectl outside the VMs in which case you need it installed.

---

## One‑command setup

From the repository root:

```bash
vagrant up
```

This will:

1. Provision a local Kubernetes cluster
2. Build ML images on the host
3. Import the Api Service and Training Service images into containerd on the worker node
4. Initialize Kubernetes components

Cluster access config will be generated locally.

**YES** - I'm a stickler for automation. Everything is automated - no step 1 run this, step 2 run that and that. I do not want to burden developers with the need to run a multitude of scripts or setup commands to use my tools.

---

## What this project demonstrates (TODO)

**ML workflow concepts**:

- Periodic / batch training (job‑style ML)
- Feature extraction & model fitting with scikit‑learn
- Artifact generation (trained model)
- Stateless inference service loading a model

**Platform / MLOps concepts**:

- Kubernetes Jobs vs long‑running Services
- Containerized ML workloads
- Image usage with `containerd` (no Docker inside the cluster)
- Local, reproducible Kubernetes via Vagrant
- Offline / air‑gapped‑friendly workflows (no registry push required)

---

## Architecture overview

High‑level flow:

```
Data Source (yfinance)
        ↓
Training Job (scikit‑learn)
        ↓
Model Artifact (saved to volume)
        ↓
Inference API (FastAPI)
        ↓
HTTP Prediction Endpoint
```

Key idea:

- **Training** is a _batch_ concern → Kubernetes `Job`
- **Inference** is a _serving_ concern → Kubernetes `Deployment + Service`

---

## Interacting with the service

From your terminal, run:

```bash
curl http://<node-ip>:30951/predictionForTomorrow
```

Example Execution:

```bash
curl http://192.168.56.10:30951/predictionForTomorrow

{"prediction":"DOWN","confidence":0.345}
```

An easy way to find the NodeIPs is via the kubectl command and go for INTERNAL-IP:

```bash
vagrant@kube-worker-1:~$ kubectl get nodes -o wide
NAME                 STATUS   ROLES           AGE   VERSION   INTERNAL-IP     EXTERNAL-IP   OS-IMAGE       KERNEL-VERSION      CONTAINER-RUNTIME
kube-control-plane   Ready    control-plane   66m   v1.31.1   192.168.56.10   <none>        Ubuntu 25.04   6.14.0-34-generic   containerd://2.2.1
kube-worker-1        Ready    <none>          65m   v1.31.1   192.168.56.21   <none>        Ubuntu 25.04   6.14.0-34-generic   containerd://2.2.1
```

---

## Some quirks

- This project is not production‑hardened

  - No auth
  - No autoscaling
  - No model registry (yet)

- You may see and verify the container registry for the application images as pointing to localhost. The project does not set up a local container registry, but it should not be an issue as long as we set _imagePullPolicy: IfNotPresent_ in our manifests.

  ```
  vagrant@kube-worker-1:~$ sudo ctr -n k8s.io images ls | grep localhost
  localhost/ml-api:latest                                                                            application/vnd.oci.image.index.v1+json                   sha256:72e63c829a4c5a60b5161855d3d67829bf990de56b8334e112c5086b662f620c 166.9 MiB linux/arm64                                                                  io.cri-containerd.image=managed
  localhost/ml-train:latest                                                                          application/vnd.oci.image.index.v1+json                   sha256:2cfd02a1f030b94880c7d5c9428c7a181f30384167c52ed1fd8920af2282cdd6 264.6 MiB linux/arm64                                                                  io.cri-containerd.image=managed
  ```

- Docker Hub rate limits may affect initial cluster networking components. (Do not vagrant destroy and vagrant up frequently)

- Kubernetes API Access Issue on macOS (Apple Silicon) with Vagrant + VirtualBox

  - When running this setup inside VirtualBox VMs on **macOS Apple Silicon**, you may encounter a situation where:

    - `kubectl` run on the **host** fails with:

    ```
    dial tcp <host-only-ip>:6443: connect: no route to host
    ```

    The problem is in the Golang net.Dial method, used by kubectl, and not the cluster setup. This is demonstrated as follows:

    ```
    % cat ./test.go
    package main
    import (
      "net"
      "fmt"
    )
    func main() {
      conn, err := net.Dial("tcp", "192.168.56.10:6443")
      if err != nil { fmt.Println("fail:", err); return }
      fmt.Println("connected", conn.RemoteAddr())
      conn.Close()
    }
    % go run ./test.go fail: dial tcp 192.168.56.10:6443: connect: no route to host
    % curl -k "https://192.168.56.10:6443/version?timeout=32s" { "major": "1", "minor": "31", "gitVersion": "v1.31.1", "gitCommit": "948afe5ca072329a73c8e79ed5938717a5cb3d21", "gitTreeState": "clean", "buildDate": "2024-09-11T21:22:08Z", "goVersion": "go1.22.6", "compiler": "gc", "platform": "linux/arm64" }%
    % kubectl get no --kubeconfig=./infra-context/admin.conf E0103 15:56:35.815332 27589 memcache.go:265] "Unhandled Error" err="couldn't get current server API group list: Get \"https://192.168.56.10:6443/api?timeout=32s\": dial tcp 192.168.56.10:6443: connect: no route to host" E0103 15:56:35.815525 27589 memcache.go:265] "Unhandled Error" err="couldn't get current server API group list: Get \"https://192.168.56.10:6443/api?timeout=32s\": dial tcp 192.168.56.10:6443: connect: no route to host" E0103 15:56:35.816710 27589 memcache.go:265] "Unhandled Error" err="couldn't get current server API group list: Get \"https://192.168.56.10:6443/api?timeout=32s\": dial tcp 192.168.56.10:6443: connect: no route to host" E0103 15:56:35.816820 27589 memcache.go:265] "Unhandled Error" err="couldn't get current server API group list: Get \"https://192.168.56.10:6443/api?timeout=32s\": dial tcp 192.168.56.10:6443: connect: no route to host" E0103 15:56:35.818031 27589 memcache.go:265] "Unhandled Error" err="couldn't get current server API group list: Get \"https://192.168.56.10:6443/api?timeout=32s\": dial tcp 192.168.56.10:6443: connect: no route to host" Unable to connect to the server: dial tcp 192.168.56.10:6443: connect: no route to host
    ```

    curl works, but not kubectl. A detailed explanation of the phenomenon is as follows:

    ### Network Topology

    Original (Failing) Setup

    ```
    +--------------------+
    | macOS Host         |
    |                    |
    | kubectl (Go)       |
    |                    |
    +---------+----------+
              |
              |  VirtualBox host-only network
              |  (192.168.56.0/24)
              |
    +---------v----------+
    | Control Plane VM   |
    |                    |
    | kube-apiserver     |
    | 0.0.0.0:6443       |
    +--------------------+
    ```

    curl and nc can reach the API server

    Go-based clients (kubectl, custom Go programs) fail with _no route to host_

    ### Root Cause

    This issue is caused by an interaction between:

    - macOS (Apple Silicon)
    - VirtualBox host-only networking
    - Go’s networking stack

    On this platform combination, VirtualBox host-only interfaces may not be
    fully visible to the routing APIs used by Go. As a result:

    - The TCP connection fails at the routing stage
    - The failure occurs before TLS or authentication
    - Other user-space networking tools may still work

    This is a host ↔ VM networking limitation, and not a Kubernetes misconfiguration.

    ### Workaround

    Port Forwarding and Localhost Access for the KubeApi Server is applied through the Vagrantfile. This is automated during provisioning with vagrant.

    Network Topology (Fixed)

    ```
    +--------------------+
    | macOS Host         |
    |                    |
    | kubectl (Go)       |
    | https://127.0.0.1  |
    +---------+----------+
              |
              |  VirtualBox NAT (port forwarding)
              |  host:6443 → guest:6443
              |
    +---------v----------+
    | Control Plane VM   |
    |                    |
    | kube-apiserver     |
    | 0.0.0.0:6443       |
    +--------------------+
    ```

    This path avoids host-only networking and is fully compatible with Go’s networking stack.
    Following this, for apple mac silicon users, we request running kubectl as follows:

    ```
    kubectl --kubeconfig=./infra-context/local-admin.conf --insecure-skip-tls-verify=true {params and args}
    ```

    The local-admin.conf is a k8s config file pointing to localhost. TLS verification needs to be skipped as localhost is not part of the advertised addresses the kube api server was configured with kubeadm. Since this is a workaround which I aim to fix cleanly in the future, I leave it as this.
    For anyone else not encountering this issue, run kubectl as follows:

    ```
    kubectl --kubeconfig=./infra-context/admin.conf {params and args}
    ```

---

## Next extensions

- Add MLflow for experiment tracking and model registration (current release only runs it on the side, with no app integration yet)
- Add another interface to the model via KServe (to compare with a regular FastAPI)
- Replace local storage with object storage (Maybe)

---
