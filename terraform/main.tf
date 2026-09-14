terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 4.0"
    }
  }
}

provider "docker" {
  # L'hôte est commenté pour utiliser la valeur par défaut du système
  # (npipe sur Windows si configuré par défaut, ou unix socket sur Linux pour CI/CD)
  # host = "npipe:////./pipe/docker_engine"
}

# ─── IMAGES ────────────────────────────────────────────────────────────────────

resource "docker_image" "minio" {
  name         = "minio/minio:latest"
  keep_locally = true
}

resource "docker_image" "postgres" {
  name         = "postgres:15"
  keep_locally = true
}

resource "docker_image" "redis" {
  name         = "redis:7"
  keep_locally = true
}

# ─── VOLUMES (stockage persistant) ─────────────────────────────────────────────

resource "docker_volume" "minio_data" {
  name = "minio_data"
}

resource "docker_volume" "postgres_data" {
  name = "postgres_data"
}

resource "docker_volume" "redis_data" {
  name = "redis_data"
}

# ─── RÉSEAU COMMUN ──────────────────────────────────────────────────────────────

resource "docker_network" "smartgrid_net" {
  name = "smartgrid_network"
}

# ─── CONTENEUR MINIO (Stockage S3 compatible) ──────────────────────────────────

resource "docker_container" "minio" {
  image   = docker_image.minio.image_id
  name    = "minio"
  restart = "unless-stopped"

  ports {
    internal = 9000
    external = 9000
  }
  ports {
    internal = 9001
    external = 9001
  }

  env = [
    "MINIO_ROOT_USER=${var.minio_access_key}",
    "MINIO_ROOT_PASSWORD=${var.minio_secret_key}"
  ]

  command = ["server", "/data", "--console-address", ":9001"]

  volumes {
    volume_name    = docker_volume.minio_data.name
    container_path = "/data"
  }

  networks_advanced {
    name = docker_network.smartgrid_net.name
  }
}

# ─── CONTENEUR POSTGRESQL (Catalogue Iceberg & Feast) ──────────────────────────

resource "docker_container" "postgres" {
  image   = docker_image.postgres.image_id
  name    = "postgres"
  restart = "unless-stopped"

  ports {
    internal = 5432
    external = 5432
  }

  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=${var.postgres_db}"
  ]

  volumes {
    volume_name    = docker_volume.postgres_data.name
    container_path = "/var/lib/postgresql/data"
  }

  networks_advanced {
    name = docker_network.smartgrid_net.name
  }
}

# ─── CONTENEUR REDIS (Feature Store Online) ────────────────────────────────────

resource "docker_container" "redis" {
  image   = docker_image.redis.image_id
  name    = "redis"
  restart = "unless-stopped"

  ports {
    internal = 6379
    external = 6379
  }

  volumes {
    volume_name    = docker_volume.redis_data.name
    container_path = "/data"
  }

  networks_advanced {
    name = docker_network.smartgrid_net.name
  }
}
