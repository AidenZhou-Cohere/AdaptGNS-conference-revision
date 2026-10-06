#!/usr/bin/env python3
"""Exact-release Sand downloader v2; separately frozen host correction, no pickle."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import http.cookiejar
import json
import math
from pathlib import Path
import re
import stat
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


DOI = "10.17603/ds2-0phb-dg64"
PORTAL = "https://www.designsafe-ci.org"
PROJECT = PORTAL + "/data/browser/public/designsafe.storage.published/PRJ-3702"
SOURCE = "/published-data/PRJ-3702/Project--graph-network-simulator-datasets/data/Sand/dataset/"
ROUTE = PORTAL + "/api/datafiles/tapis/public/download/designsafe.storage.published/?doi=" + urllib.parse.quote(DOI, safe="")
FILES = {"metadata.json": 363, "train.npz": 2676940383,
         "valid.npz": 82712898, "test.npz": 85825802}
METADATA_SHA256 = "cef268faed1e0c3265f92395c364bddaff523bc8c117beae050de122f88059a0"
PORTAL_HOSTS = {"www.designsafe-ci.org"}
DOWNLOAD_HOSTS = {"www.designsafe-ci.org", "designsafe.tapis.io", "designsafe-download01.tacc.utexas.edu"}
MAX_HEADER = 65536
MAX_MEMBERS = 4096
MAX_UNCOMPRESSED = 32 * 1024 ** 3
SAFE_HEADERS = ("Content-Length", "Content-Type", "Content-Encoding", "ETag", "Last-Modified")


class AdmissionError(ValueError):
    """Only fixed, non-sensitive reason strings belong in this exception."""


def require(condition, reason):
    if not condition:
        raise AdmissionError(reason)


def check_url(url, hosts):
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == "https" and parsed.hostname in hosts
            and parsed.username is None and parsed.password is None
            and parsed.port in (None, 443) and not parsed.fragment,
            "URL is outside the fixed HTTPS host policy")
    return parsed.hostname


class LimitedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts):
        self.hosts, self.count = hosts, 0

    def redirect_request(self, request, response, code, message, headers, new_url):
        check_url(new_url, self.hosts)
        self.count += 1
        require(self.count <= 3, "Redirect limit exceeded")
        return super().redirect_request(request, response, code, message, headers, new_url)


def opener(hosts, cookies=None):
    handlers = [urllib.request.ProxyHandler({}), LimitedRedirect(hosts)]
    if cookies is not None:
        handlers.append(urllib.request.HTTPCookieProcessor(cookies))
    return urllib.request.build_opener(*handlers)


def remaining(deadline):
    seconds = deadline - time.monotonic()
    require(seconds > 0, "Acquisition time limit exceeded")
    return min(30.0, seconds)


def open_response(client, request, hosts, deadline):
    check_url(request.full_url, hosts)
    response = client.open(request, timeout=remaining(deadline))
    try:
        check_url(response.geturl(), hosts)
        require(response.status == 200, "Unexpected HTTP success status")
        require(response.headers.get("Content-Encoding", "identity").lower() == "identity",
                "Unexpected HTTP content encoding")
    except Exception:
        response.close()
        raise
    return response


def small_body(response, deadline, limit):
    chunks, size = [], 0
    while True:
        remaining(deadline)
        block = response.read(min(8192, limit + 1 - size))
        if not block:
            return b"".join(chunks)
        chunks.append(block)
        size += len(block)
        require(size <= limit, "Portal response exceeded size limit")


def acquire_link(portal, cookies, name, deadline):
    tokens = [cookie.value for cookie in cookies if cookie.name == "csrftoken"
              and cookie.domain.lstrip(".") in {"www.designsafe-ci.org", "designsafe-ci.org"}]
    require(len(tokens) == 1 and re.fullmatch(r"[A-Za-z0-9]{32,64}", tokens[0]) is not None,
            "No unique normal portal CSRF cookie; authentication was not attempted")
    request = urllib.request.Request(ROUTE,
        data=json.dumps({"paths": [SOURCE + name]}).encode("utf-8"), method="PUT",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Accept-Encoding": "identity", "X-CSRFToken": tokens[0],
                 "Origin": PORTAL, "Referer": PROJECT})
    with open_response(portal, request, PORTAL_HOSTS, deadline) as response:
        body = json.loads(small_body(response, deadline, MAX_HEADER))
    require(isinstance(body, dict) and isinstance(body.get("href"), str),
            "Portal did not return a download href")
    check_url(body["href"], DOWNLOAD_HOSTS)
    return body["href"]


def dtype_layout(description, depth=0):
    """Parse NPY dtype descriptors as literals, never through pickle or NumPy."""
    require(depth <= 8, "Nested dtype exceeds inspection limit")
    if isinstance(description, str):
        match = re.fullmatch(r"[<>=|]?([?biufcOSUVmM])([0-9]*)(?:\[[A-Za-z0-9]+\])?", description)
        require(match is not None, "Unsupported NPY dtype descriptor")
        kind, width = match.groups()
        if kind == "O":
            return True, None
        if kind == "?":
            require(not width, "Malformed boolean dtype")
            return False, 1
        require(bool(width), "NPY dtype is missing its width")
        size = int(width) * (4 if kind == "U" else 1)
        require(size <= MAX_UNCOMPRESSED, "NPY dtype width exceeds inspection limit")
        return False, size
    require(isinstance(description, list), "Unsupported structured NPY descriptor")
    total, has_object = 0, False
    for field in description:
        require(isinstance(field, tuple) and len(field) in (2, 3), "Malformed structured NPY field")
        objects, size = dtype_layout(field[1], depth + 1)
        shape = () if len(field) == 2 else field[2]
        require(isinstance(shape, tuple) and len(shape) <= 8
                and all(type(value) is int and value >= 0 for value in shape),
                "Malformed NPY subarray shape")
        has_object |= objects
        total += (size or 0) * math.prod(shape)
        require(total <= MAX_UNCOMPRESSED, "Structured dtype exceeds inspection limit")
    return has_object, None if has_object else total


def npy_header(stream, member_size):
    require(stream.read(6) == b"\x93NUMPY", "ZIP member is not an NPY file")
    version = stream.read(2)
    require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Unsupported NPY format version")
    width = 2 if version[0] == 1 else 4
    raw_length = stream.read(width)
    require(len(raw_length) == width, "Truncated NPY header length")
    size = struct.unpack("<H" if width == 2 else "<I", raw_length)[0]
    require(0 < size <= MAX_HEADER and 8 + width + size <= member_size, "Invalid or oversized NPY header")
    raw = stream.read(size)
    require(len(raw) == size, "Truncated NPY header")
    header = ast.literal_eval(raw.decode("utf-8" if version[0] == 3 else "latin1"))
    require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"},
            "Unexpected NPY header fields")
    shape = header["shape"]
    require(isinstance(shape, tuple) and len(shape) <= 8
            and all(type(value) is int and value >= 0 for value in shape)
            and type(header["fortran_order"]) is bool, "Malformed NPY shape/order")
    objects, itemsize = dtype_layout(header["descr"])
    payload_bytes = member_size - (8 + width + size)
    if not objects:
        require(math.prod(shape) * itemsize == payload_bytes, "Numeric NPY payload length disagrees with header")
    else:
        require(payload_bytes > 0, "Object NPY has no payload")
    return {"version": list(version), "shape": list(shape), "descr": header["descr"],
            "fortran_order": header["fortran_order"], "has_object_dtype": objects,
            "header_bytes": 8 + width + size, "payload_bytes": payload_bytes,
            "payload_decoded": False}


def zip_inventory(path, deadline):
    rows, total = [], 0
    # Bound the directory before ZipFile materializes its member inventory.
    archive_bytes = path.stat().st_size
    with path.open("rb") as source:
        tail_offset = max(0, archive_bytes - 65557)
        source.seek(tail_offset)
        tail = source.read(65557)
    end = tail.rfind(b"PK\x05\x06")
    require(end >= 0 and end + 22 <= len(tail), "Missing ZIP end record")
    _, disk, directory_disk, disk_count, count, directory_bytes, directory_offset, comment_bytes = struct.unpack(
        "<4s4H2IH", tail[end:end + 22])
    locator_offset = tail_offset + end - 20
    if locator_offset >= 0:
        with path.open("rb") as source:
            source.seek(locator_offset)
            locator = source.read(20)
            if locator[:4] == b"PK\x06\x07":
                _, locator_disk, record_offset, disks = struct.unpack("<4sIQI", locator)
                require(locator_disk == 0 and disks == 1 and 0 <= record_offset <= locator_offset - 56,
                        "Unsupported ZIP64 locator")
                source.seek(record_offset)
                raw_record = source.read(56)
                require(len(raw_record) == 56, "Truncated ZIP64 end record")
                signature, record_size, _, _, disk, directory_disk, disk_count, count, directory_bytes, directory_offset = struct.unpack(
                    "<4sQ2H2I4Q", raw_record)
                require(signature == b"PK\x06\x06" and record_size == 44
                        and record_offset + 12 + record_size == locator_offset,
                        "Unsupported ZIP64 end record")
    require(end + 22 + comment_bytes == len(tail) and disk == directory_disk == 0
            and disk_count == count and 0 < count <= MAX_MEMBERS
            and directory_bytes <= 8 * 1024 ** 2
            and directory_offset + directory_bytes <= archive_bytes,
            "Unsupported or oversized ZIP directory")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) == count, "Unexpected ZIP member count")
        require(len({item.filename for item in members}) == len(members), "Duplicate ZIP member names")
        for index, member in enumerate(members):
            remaining(deadline)
            mode = stat.S_IFMT(member.external_attr >> 16)
            require(re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.npy", member.filename) is not None
                    and not member.is_dir() and mode in (0, stat.S_IFREG), "Unsafe or unexpected ZIP member")
            require(not member.flag_bits & 1 and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    "Encrypted or unsupported ZIP member")
            total += member.file_size
            require(0 < member.file_size <= MAX_UNCOMPRESSED and total <= MAX_UNCOMPRESSED,
                    "ZIP uncompressed size exceeds inspection limit")
            with archive.open(member) as stream:
                header = npy_header(stream, member.file_size)
            rows.append({"source_member_index": index, "name": member.filename,
                         "compressed_bytes": member.compress_size, "uncompressed_bytes": member.file_size,
                         "listed_crc32": f"{member.CRC:08x}", "npy": header})
    return {"member_count": len(rows), "uncompressed_bytes": total, "members": rows,
            "full_member_crc_verified": False, "inspection": "ZIP inventory and NPY headers only; no array payload decoding",
            "object_dtypes_detected": any(row["npy"]["has_object_dtype"] for row in rows)}


def safe_error(error):
    result = {"error_type": type(error).__name__}
    if isinstance(error, AdmissionError):
        result["reason"] = str(error)
    if isinstance(error, urllib.error.HTTPError):
        result["http_status"] = error.code
    return result  # No raw response, exception URL, cookie, or capability token.


def save_report(directory, report):
    temporary = directory / "download_report.json.tmp"
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    temporary.replace(directory / "download_report.json")


def download_file(portal, cookies, name, directory, deadline, record):
    record["stage"] = "acquire_download_link"
    href = acquire_link(portal, cookies, name, deadline)
    record["download_host"] = check_url(href, DOWNLOAD_HOSTS)
    # A separate opener has no CookieJar, Authorization, CSRF or Referer header.
    client = opener(DOWNLOAD_HOSTS)
    request = urllib.request.Request(href, headers={"Accept-Encoding": "identity"})
    partial = directory / (name + ".part")
    digest, received = hashlib.sha256(), 0
    record["stage"] = "download"
    try:
        with open_response(client, request, DOWNLOAD_HOSTS, deadline) as response, partial.open("xb") as output:
            record["response_headers"] = {key: response.headers[key] for key in SAFE_HEADERS if key in response.headers}
            record["final_download_host"] = check_url(response.geturl(), DOWNLOAD_HOSTS)
            length = response.headers.get("Content-Length")
            require(length is None or int(length) == FILES[name], "HTTP Content-Length differs from the published listing")
            while True:
                remaining(deadline)
                block = response.read(min(1024 ** 2, FILES[name] - received + 1))
                if not block:
                    break
                output.write(block)
                digest.update(block)
                received += len(block)
                require(received <= FILES[name], "Download exceeded the published byte count")
    finally:
        record.update(received_bytes=received, received_sha256=digest.hexdigest(), partial_name=partial.name)
    require(received == FILES[name], "Download ended before the published byte count")
    record["stage"] = "structural_inspection"
    if name == "metadata.json":
        require(digest.hexdigest() == METADATA_SHA256, "Metadata differs from the verified public/local bytes")
        record["metadata_matches_verified_bytes"] = True
    else:
        record["zip"] = zip_inventory(partial, deadline)
    partial.replace(directory / name)
    record.update(status="downloaded_and_inspected", stage="complete", saved_name=name)
    record.pop("partial_name", None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True, help="New directory; its parent must already exist")
    parser.add_argument("--files", nargs="+", choices=FILES, default=list(FILES))
    parser.add_argument("--max-seconds", type=int, default=1800, help="Whole-invocation time cap, default1800")
    args = parser.parse_args(argv)
    if len(set(args.files)) != len(args.files) or not 1 <= args.max_seconds <= 7200:
        parser.error("Unique filenames and max-seconds in1..7200 are required")
    names = ["metadata.json"] + [name for name in args.files if name != "metadata.json"]
    try:
        args.output.mkdir(mode=0o700, exist_ok=False)
    except OSError as error:
        print(json.dumps({"status": "not_started", **safe_error(error)}))
        return 1
    started = time.monotonic()
    deadline = started + args.max_seconds
    report = {"schema": "sand_public_acquisition_v2", "status": "in_progress", "doi": DOI,
              "project_id": "PRJ-3702", "version": 1, "dataset": "Sand", "published_directory": SOURCE,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "downloader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": {"max_seconds": args.max_seconds, "socket_timeout_seconds": 30,
                         "redirects_per_opener": 3, "max_npy_header_bytes": MAX_HEADER,
                         "max_zip_members": MAX_MEMBERS, "max_uncompressed_bytes": MAX_UNCOMPRESSED},
              "metadata_reference_sha256": METADATA_SHA256, "requested_files": names,
              "publisher_npz_sha256_available": False, "arrays_decoded": False, "pickle_used": False,
              "conversion_performed": False, "metrics_computed": False,
              "files": [{"name": name, "expected_bytes": FILES[name], "status": "not_attempted"} for name in names]}
    save_report(args.output, report)
    try:
        cookies = http.cookiejar.CookieJar()
        portal = opener(PORTAL_HOSTS, cookies)
        request = urllib.request.Request(PROJECT, headers={"Accept-Encoding": "identity"})
        with open_response(portal, request, PORTAL_HOSTS, deadline) as response:
            small_body(response, deadline, 1024 ** 2)
        for record in report["files"]:
            record["status"] = "in_progress"
            save_report(args.output, report)
            try:
                download_file(portal, cookies, record["name"], args.output, deadline, record)
            except Exception as error:
                record.update(status="failed", **safe_error(error))
                raise
            finally:
                save_report(args.output, report)
        objects = any(record.get("zip", {}).get("object_dtypes_detected") for record in report["files"])
        report["status"] = "complete_with_object_dtypes" if objects else "complete_structural_only"
        report["object_dtypes_detected"] = objects
        report["numeric_payload_validity_verified"] = False
    except Exception as error:
        report.update(status="failed", failure=safe_error(error))
    report["elapsed_seconds"] = time.monotonic() - started
    save_report(args.output, report)
    print(json.dumps({"status": report["status"], "report_name": "download_report.json",
                      "arrays_decoded": False, "pickle_used": False}))
    return 1 if report["status"] == "failed" else 2 if report["object_dtypes_detected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
