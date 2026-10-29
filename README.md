# GCP Slurm Cluster with Cluster Toolkit

A minimal [Google Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit) blueprint that deploys an autoscaling Slurm cluster on Google Cloud.

## What it builds

| Module | Purpose |
|---|---|
| `network` | VPC for the cluster |
| `homefs` | 1 TiB `BASIC_HDD` Filestore share mounted as `/home` on every node |
| `compute_nodeset` | Up to 10 `c2-standard-60` compute VMs, created on demand and deleted when idle |
| `compute_partition` | The `compute` Slurm partition (default queue) |
| `slurm_login` | Login node where users SSH in and run `sbatch` / `srun` (private IP only) |
| `slurm_controller` | Slurm controller that ties everything together (private IP only) |

The VPC module also creates firewall rules for internal traffic and IAP SSH, plus a Cloud Router and Cloud NAT so private VMs can reach the internet outbound.

## How it works

You don't write Terraform by hand. The `gcluster` CLI reads the YAML blueprint, resolves the `use:` links between modules, generates a full Terraform project, and runs it.

```
blueprint (YAML)  →  gcluster  →  Terraform (+ Packer)  →  cluster on GCP
```

## Prerequisites

- A Google Cloud project with billing enabled
- `gcloud` authenticated (`gcloud auth application-default login`)
- [Terraform](https://developer.hashicorp.com/terraform/install) installed
- The `gcluster` binary, from the [Cluster Toolkit releases](https://github.com/GoogleCloudPlatform/cluster-toolkit/releases) or built from source with `make`

### Permissions and APIs (one-time)

Run `setup-permissions.sh` as a project Owner or Project IAM Admin. It enables the required APIs, grants the deployer the roles to create the cluster and SSH in through IAP, and grants the default Compute Engine service account the roles the Slurm controller needs to create compute VMs.

```bash
./setup-permissions.sh <YOUR_PROJECT_ID> <DEPLOYER_EMAIL>
# Already a project Owner? Skip the deploy roles:
SKIP_USER_ROLES=1 ./setup-permissions.sh <YOUR_PROJECT_ID> <DEPLOYER_EMAIL>
```

Also check quotas: 10 `c2-standard-60` nodes need 600 C2 vCPUs in the region.

## Usage

```bash
# 1. Generate the Terraform deployment folder
./gcluster create hpc-slurm.yaml --vars project_id=<YOUR_PROJECT_ID>

# 2. Deploy (runs terraform init/plan/apply)
./gcluster deploy hpc-slurm

# 3. SSH to the login node through IAP (it has no public IP)
gcloud compute instances list --filter="name~login"   # find the login VM name
gcloud compute ssh <LOGIN_VM_NAME> --zone us-central1-a --tunnel-through-iap

# 4. Wait until `sinfo` shows the compute partition, then submit a test job
srun -N 2 hostname

# 5. Tear everything down when finished
./gcluster destroy hpc-slurm
```

## Using plain Terraform instead

Run `gcluster create` once, then manage the generated `hpc-slurm/` folder with normal `terraform plan` / `terraform apply`, a remote state backend and your usual CI.

## Customizing

- **GPUs:** change `machine_type` in `compute_nodeset` to a GPU type such as `a3-highgpu-8g` (check quota and zone availability).
- **Cluster size:** adjust `node_count_dynamic_max`.
- **Region/zone:** edit the `vars` block.
- **Faster storage:** set `filestore_tier: BASIC_SSD` and `size_gb: 2560` (the SSD tier's minimum).

## Notes

- Module paths and settings change between toolkit releases. Compare against the `examples/` folder of the version you install, especially `examples/hpc-slurm.yaml`.
- The login and controller nodes have no public IPs. SSH access uses IAP, which needs the `IAP-secured Tunnel User` role on your account.
- Running this creates billable resources. Remember to `destroy` when done.
