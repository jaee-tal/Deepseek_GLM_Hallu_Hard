# GCP runner setup

The application itself is portable. On GCP, use one `a4-highgpu-8g` VM per active
model, a service account with Cloud Storage object and Monitoring metric-writer
permissions, and the container built from the bundle's `Dockerfile`.

Create the email alert policies once:

```bash
cd infra/gcp
terraform init
terraform apply -var project_id="$GOOGLE_CLOUD_PROJECT" \
  -var alert_email="$CROSS_PROBE_ALERT_EMAIL"
```

The logging agent must collect container stdout. Alerts are emitted as structured
JSON and are also sent through W&B. Store `OPENROUTER_API_KEY`, `WANDB_API_KEY`,
and `HF_TOKEN` in Secret Manager and expose them to the container as environment
variables. Do not bake credentials into the image.
