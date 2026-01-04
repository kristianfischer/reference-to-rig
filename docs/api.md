# API Documentation

Reference-to-Rig Audio Engine REST API running on `http://localhost:8000`.

## Overview

The API follows REST conventions and returns JSON responses. All endpoints are prefixed with the appropriate resource path.

## Authentication

No authentication required for local use. CORS is configured to allow requests from the desktop UI.

## Endpoints

### Health Check

```
GET /health
```

Returns engine status and version.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

---

### Projects

#### Create Project

```
POST /projects
```

Create a new tone matching project.

**Request Body:**
```json
{
  "name": "My SRV Tone",
  "description": "Optional description of the target tone"
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "My SRV Tone",
  "description": "Optional description of the target tone",
  "status": "created",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "source_audio_path": null,
  "isolated_audio_path": null,
  "isolation_confidence": null
}
```

#### List Projects

```
GET /projects
```

Get all projects, sorted by most recently updated.

**Response:** `200 OK`
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "My SRV Tone",
    "status": "matched",
    ...
  }
]
```

#### Get Project

```
GET /projects/{id}
```

Get details of a specific project.

**Response:** `200 OK` or `404 Not Found`

#### Delete Project

```
DELETE /projects/{id}
```

Delete a project and all associated files.

**Response:** `204 No Content` or `404 Not Found`

---

### Audio Import

#### Import Audio File

```
POST /projects/{id}/import
Content-Type: multipart/form-data
```

Upload an audio file to the project.

**Form Data:**
- `file`: Audio file (WAV, MP3, FLAC, OGG, M4A)

**Response:** `200 OK`
```json
{
  "project_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_audio_path": "/data/projects/.../source.wav",
  "duration_seconds": 180.5,
  "sample_rate": 48000,
  "channels": 2
}
```

#### Get Audio File

```
GET /projects/{id}/audio/{type}
```

Stream an audio file from the project.

**Path Parameters:**
- `type`: One of `source`, `isolated`, `match_balanced`, `match_brighter`, `match_thicker`

**Response:** Audio file stream

---

### Processing

#### Start Guitar Isolation

```
POST /projects/{id}/isolate
```

Start background task to isolate guitar from the source audio.

**Prerequisites:**
- Project must have imported audio (`status: imported`)

**Response:** `200 OK`
```json
{
  "project_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_id": "task_abc123",
  "status": "pending"
}
```

#### Start Tone Matching

```
POST /projects/{id}/match
```

Start background task to find matching captures and optimize EQ.

**Prerequisites:**
- Project must have isolated audio (`status: isolated`)

**Response:** `200 OK`
```json
{
  "project_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_id": "task_xyz789",
  "status": "pending"
}
```

#### Get Task Status

```
GET /projects/{id}/tasks/{task_id}
```

Poll the status of a background task.

**Response:** `200 OK`
```json
{
  "task_id": "task_abc123",
  "status": "running",
  "progress": 0.45,
  "message": "Isolating guitar...",
  "result": null,
  "error": null
}
```

**Status Values:**
- `pending`: Task queued
- `running`: Task in progress
- `completed`: Task finished successfully
- `failed`: Task failed (check `error` field)

---

### Results

#### Get Match Results

```
GET /projects/{id}/results
```

Get the matching results including all candidates.

**Response:** `200 OK`
```json
{
  "project_id": "550e8400-e29b-41d4-a716-446655440000",
  "reference_audio_path": "/data/projects/.../source.wav",
  "isolated_audio_path": "/data/projects/.../isolated_guitar.wav",
  "isolation_confidence": 0.87,
  "candidates": [
    {
      "flavor": "balanced",
      "nam_model_id": "nam_001",
      "nam_model_name": "Clean Fender Twin",
      "ir_id": "ir_002",
      "ir_name": "Jensen P12R 1x12",
      "input_gain_db": 2.5,
      "eq_settings": {
        "bands": [
          {
            "frequency": 100,
            "gain_db": -1.5,
            "q": 0.7,
            "band_type": "lowshelf"
          },
          ...
        ],
        "highpass_freq": 80,
        "lowpass_freq": 12000
      },
      "similarity_score": 0.82,
      "rendered_audio_path": "/data/projects/.../rendered_balanced.wav"
    },
    ...
  ],
  "created_at": "2024-01-15T10:35:00Z"
}
```

---

### Export

#### Export JSON Preset

```
GET /projects/{id}/export/json
```

Get the rig recipe as JSON for programmatic use.

**Response:** `200 OK`
```json
{
  "version": "1.0",
  "project_name": "My SRV Tone",
  "created_at": "2024-01-15T10:35:00Z",
  "reference_description": "Texas Flood intro tone",
  "candidates": [...],
  "notes": null
}
```

#### Export Markdown Recipe

```
GET /projects/{id}/export/markdown
```

Download the rig recipe as a human-readable Markdown file.

**Response:** Markdown file download

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Status Codes:**
- `400 Bad Request`: Invalid input or precondition not met
- `404 Not Found`: Resource doesn't exist
- `500 Internal Server Error`: Server-side error
- `501 Not Implemented`: Feature not available in MVP

---

## WebSocket (Future)

Real-time task progress updates via WebSocket are planned for a future release. Currently, poll the task status endpoint.


