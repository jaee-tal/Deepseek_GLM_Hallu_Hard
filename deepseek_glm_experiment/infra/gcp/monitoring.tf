terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 7.0" }
  }
}

variable "project_id" { type = string }
variable "alert_email" { type = string }

provider "google" { project = var.project_id }

resource "google_monitoring_metric_descriptor" "runner" {
  for_each = {
    heartbeat             = "Runner heartbeat"
    idle_seconds          = "Seconds since filesystem progress"
    checkpoint_age_seconds = "Seconds since the last verified checkpoint"
    disk_used_fraction    = "Runner disk fraction used"
  }
  type         = "custom.googleapis.com/cross_probe/${each.key}"
  display_name = each.value
  description  = each.value
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  dynamic "labels" {
    for_each = toset(["experiment_id", "model", "stage"])
    content {
      key         = labels.value
      value_type  = "STRING"
      description = labels.value
    }
  }
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "Cross-probe experiment email"
  type         = "email"
  labels       = { email_address = var.alert_email }
}

resource "google_logging_metric" "experiment_alert" {
  name   = "cross_probe_experiment_alerts"
  filter = "jsonPayload.event=\"alert\" AND jsonPayload.experiment_id:*"
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_monitoring_alert_policy" "process_alert" {
  display_name = "Cross-probe process failure or stall alert"
  combiner     = "OR"
  conditions {
    display_name = "Structured experiment alert logged"
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.experiment_alert.name}\" AND resource.type=\"gce_instance\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.name]
}

resource "google_monitoring_alert_policy" "heartbeat_missing" {
  display_name = "Cross-probe heartbeat missing for 15 minutes"
  combiner     = "OR"
  conditions {
    display_name = "No runner heartbeat"
    condition_absent {
      filter   = "metric.type=\"${google_monitoring_metric_descriptor.runner["heartbeat"].type}\" AND resource.type=\"global\""
      duration = "900s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.name]
}

resource "google_monitoring_alert_policy" "checkpoint_stale" {
  display_name = "Cross-probe checkpoint older than 90 minutes"
  combiner     = "OR"
  conditions {
    display_name = "Checkpoint age"
    condition_threshold {
      filter          = "metric.type=\"${google_monitoring_metric_descriptor.runner["checkpoint_age_seconds"].type}\" AND resource.type=\"global\""
      comparison      = "COMPARISON_GT"
      threshold_value = 5400
      duration        = "300s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.name]
}

resource "google_monitoring_alert_policy" "stage_idle" {
  display_name = "Cross-probe stage idle for 15 minutes"
  combiner     = "OR"
  conditions {
    display_name = "No experiment file progress"
    condition_threshold {
      filter          = "metric.type=\"${google_monitoring_metric_descriptor.runner["idle_seconds"].type}\" AND resource.type=\"global\""
      comparison      = "COMPARISON_GT"
      threshold_value = 900
      duration        = "60s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.name]
}

resource "google_monitoring_alert_policy" "disk_full" {
  display_name = "Cross-probe disk utilization above 85 percent"
  combiner     = "OR"
  conditions {
    display_name = "Runner disk utilization"
    condition_threshold {
      filter          = "metric.type=\"${google_monitoring_metric_descriptor.runner["disk_used_fraction"].type}\" AND resource.type=\"global\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0.85
      duration        = "300s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.name]
}
