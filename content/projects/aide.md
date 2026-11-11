+++
title = 'Artificial Intelligence Development Environment (AIDE)'
date = 2023-01-01
until = '2025'
description = 'Architect and technical lead for AIDE, a batteries-included platform for collaborative data science, machine learning, and model deployment spanning the MLOps lifecycle.'
org = 'Artificial Intelligence Integration Center'

[[links]]
  label = 'site'
  url = 'https://aide.ai.army.mil/'
+++

I served as architect and technical lead for AIDE during my final ~2 years at the Artificial Intelligence Integration Center. The platform covers the complete MLOps lifecycle, offering comparable features to [Kubeflow](https://www.kubeflow.org/) with less modularity. It is available to DoD users on NIPR.

AIDE gives teams one place to manage source code, data, and compute for interactive data science and model building. Beyond the interactive workflow, it automates application deployment, model endpoint deployment, and job and pipeline scheduling.

The platform is cloud-native, deployed on Docker and Kubernetes, and integrates [Istio](https://istio.io/) for service mesh, [Keycloak](https://www.keycloak.org/) for authentication, [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) for monitoring, and [Jaeger](https://www.jaegertracing.io/) for tracing. The codebase spans Go, TypeScript, and Python and totals over 500,000 source lines of code.
