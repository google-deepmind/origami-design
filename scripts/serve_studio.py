# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Local COrigami Studio & Editor dev server and PEP 427 wheel builder.

Serves `docs/` (COrigami Studio at `/` and COrigami WASM Editor at `/editor/`)
over HTTP while dynamically packaging `src/bp_pipeline/` into
`/py/bp_pipeline-0.1.0-py3-none-any.whl` in memory (no binary `.whl` artifacts
written to the repository). Also supports `--build-wheel <path>` for static
GitHub Pages CI builds.
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import http.server
import io
import pathlib
import socket
import socketserver
import urllib.parse
import zipfile

WHEEL_FILENAME = 'bp_pipeline-0.1.0-py3-none-any.whl'
WHEEL_URL_PATH = f'/py/{WHEEL_FILENAME}'
_REPO_ROOT = pathlib.Path(__file__).parents[1]
_SKIP_NAMES = frozenset({'__pycache__', 'BUILD', 'OWNERS', '.DS_Store'})
_SKIP_SUFFIXES = frozenset({'.pyc', '.pyo', '.pyd', '.so'})
_REPRODUCIBLE_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def _urlsafe_sha256(data: bytes) -> str:
  digest = hashlib.sha256(data).digest()
  return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')


def build_wheel_bytes(
    repo_root: pathlib.Path = _REPO_ROOT,
    version: str = '0.1.0',
) -> bytes:
  """Builds a deterministic PEP 427 pure-Python wheel for `bp_pipeline`."""
  pkg_dir = repo_root / 'src' / 'bp_pipeline'
  if not pkg_dir.is_dir():
    raise FileNotFoundError(
        f'bp_pipeline source directory not found: {pkg_dir}'
    )

  dist_info = f'bp_pipeline-{version}.dist-info'
  records: list[str] = []
  buf = io.BytesIO()

  def _add_bytes(arcname: str, payload: bytes) -> None:
    zinfo = zipfile.ZipInfo(arcname, date_time=_REPRODUCIBLE_ZIP_TIME)
    zinfo.compress_type = zipfile.ZIP_DEFLATED
    zinfo.external_attr = 0o644 << 16
    zf.writestr(zinfo, payload)
    digest = _urlsafe_sha256(payload)
    records.append(f'{arcname},sha256={digest},{len(payload)}')

  with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    all_files = sorted(p for p in pkg_dir.rglob('*') if p.is_file())
    for file_path in all_files:
      rel_parts = file_path.relative_to(pkg_dir).parts
      if any(part in _SKIP_NAMES for part in rel_parts):
        continue
      if file_path.name in _SKIP_NAMES or file_path.suffix in _SKIP_SUFFIXES:
        continue
      arcname = ('bp_pipeline/' + '/'.join(rel_parts))
      _add_bytes(arcname, file_path.read_bytes())

    metadata = (
        'Metadata-Version: 2.1\n'
        'Name: bp-pipeline\n'
        f'Version: {version}\n'
        'Summary: Automated Box-Pleating Origami Pipeline for COrigami Studio\n'
        'License: Apache-2.0\n'
        'Requires-Python: >=3.12\n'
    ).encode('utf-8')
    wheel_meta = (
        'Wheel-Version: 1.0\n'
        'Generator: corigami-studio-wheel-builder\n'
        'Root-Is-Purelib: true\n'
        'Tag: py3-none-any\n'
    ).encode('utf-8')
    _add_bytes(f'{dist_info}/METADATA', metadata)
    _add_bytes(f'{dist_info}/WHEEL', wheel_meta)

    records.append(f'{dist_info}/RECORD,,')
    record_bytes = ('\n'.join(records) + '\n').encode('utf-8')
    zinfo = zipfile.ZipInfo(
        f'{dist_info}/RECORD', date_time=_REPRODUCIBLE_ZIP_TIME
    )
    zinfo.compress_type = zipfile.ZIP_DEFLATED
    zinfo.external_attr = 0o644 << 16
    zf.writestr(zinfo, record_bytes)

  return buf.getvalue()


class StudioRequestHandler(http.server.SimpleHTTPRequestHandler):
  """Serves `docs/` plus in-memory `/py/bp_pipeline-0.1.0-py3-none-any.whl`."""

  def __init__(self, *args, repo_root: pathlib.Path = _REPO_ROOT, **kwargs):
    self._repo_root = repo_root
    super().__init__(*args, directory=str(repo_root / 'docs'), **kwargs)

  def end_headers(self) -> None:
    self.send_header(
        'Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0'
    )
    self.send_header('Pragma', 'no-cache')
    self.send_header('Expires', '0')
    super().end_headers()

  def _strip_conditional_headers(self) -> None:
    for hdr in ('If-Modified-Since', 'If-None-Match'):
      if hdr in self.headers:
        del self.headers[hdr]

  def _maybe_serve_wheel(self, head_only: bool) -> bool:
    parsed = urllib.parse.urlsplit(self.path)
    if parsed.path != WHEEL_URL_PATH:
      return False
    payload = build_wheel_bytes(self._repo_root)
    self.send_response(200)
    self.send_header('Content-Type', 'application/zip')
    self.send_header('Content-Length', str(len(payload)))
    self.end_headers()
    if not head_only:
      self.wfile.write(payload)
    return True

  def do_HEAD(self) -> None:  # pylint: disable=invalid-name
    self._strip_conditional_headers()
    if self._maybe_serve_wheel(head_only=True):
      return
    super().do_HEAD()

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    self._strip_conditional_headers()
    try:
      if self._maybe_serve_wheel(head_only=False):
        return
      super().do_GET()
    except (BrokenPipeError, ConnectionResetError):
      pass


class _ReusableTCPServer(socketserver.ThreadingTCPServer):
  allow_reuse_address = True
  daemon_threads = True
  request_queue_size = 128


class _DualStackTCPServer(_ReusableTCPServer):
  """Listens on both IPv6 and IPv4 when bound to `::`."""

  address_family = socket.AF_INET6

  def server_bind(self) -> None:
    self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    super().server_bind()


def create_server(
    host: str = '127.0.0.1',
    port: int = 8080,
    repo_root: pathlib.Path = _REPO_ROOT,
) -> socketserver.ThreadingTCPServer:
  handler = functools.partial(StudioRequestHandler, repo_root=repo_root)
  if ':' in host:
    return _DualStackTCPServer((host, port), handler)
  return _ReusableTCPServer((host, port), handler)


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--build-wheel',
      type=pathlib.Path,
      default=None,
      help='Write the bp_pipeline wheel to this path and exit.',
  )
  parser.add_argument('--host', default='127.0.0.1', help='Bind address.')
  parser.add_argument('--port', type=int, default=8080, help='Bind port.')
  return parser


def main() -> None:
  args = _build_arg_parser().parse_args()

  if args.build_wheel is not None:
    wheel_bytes = build_wheel_bytes(_REPO_ROOT)
    args.build_wheel.parent.mkdir(parents=True, exist_ok=True)
    args.build_wheel.write_bytes(wheel_bytes)
    print(f'Wrote {args.build_wheel} ({len(wheel_bytes):,} bytes)')
    return

  with create_server(args.host, args.port, _REPO_ROOT) as httpd:
    actual_host, actual_port = httpd.server_address[:2]
    print(
        f'COrigami Studio: http://{actual_host}:{actual_port}/\n'
        f'COrigami Editor: http://{actual_host}:{actual_port}/editor/\n'
        f'(dynamic Pyodide wheel at {WHEEL_URL_PATH})'
    )
    httpd.serve_forever()


if __name__ == '__main__':
  main()
