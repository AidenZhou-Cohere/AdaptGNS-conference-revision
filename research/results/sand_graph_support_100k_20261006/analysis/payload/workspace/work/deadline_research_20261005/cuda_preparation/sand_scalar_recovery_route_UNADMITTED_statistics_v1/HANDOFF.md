# Sand same-phase scalar recovery route candidate

This is a separately versioned, unlaunched operational candidate. The original route package `292cc31d54e822d6eb1a843da3ce55427f9f89201c1fb9a2dd7293d0660dfd80` is unchanged. No scientific worker, model, data, phase or remote action has been executed by this preparation.

The old route reached its accepted-connection cap of 128. Its source did not count cap-rejected clients, so that observation does not identify the cause of the Go panic or later EOF. Root preserved original session 23611 exit 130 after SIGINT to exact singleton PID 46711, the terminal with zero worker threads, and independent local absence of PIDs/groups 46711, 47926 and 48109. The evidence review and hashes are in `closure_evidence_review.json`.

## Changes and fixed limits

`phase_binding.py` is byte-identical to the original. Every use still reads the exact original phase SHA `0ac7a368dd8afa921c8b973c5a8a760af7f10eb9f3413bceb592fff0c0ad7a5c` and anchor SHA `06cee5e7a2372fb5799678cfa9493f91a0e88fc3ba45ebc1a70bc15944295fd1`. The original stop remains 2026-10-06T23:33:01.226132+00:00; launch cannot reset it or grant another hour.

Only exact CONNECT to `coder.internal.cohere.com:443` is forwarded opaquely to the previously verified `100.106.33.61:443`. The route binds loopback, uses backlog 4, at most eight handler threads and a finite lifetime cap of 2,048 accepted clients. Separate counters record cap, concurrency and non-loopback rejection. No subprocesses, credentials, global proxy changes, content logs or TLS interception are introduced.

Payload forwarding is now nonblocking with at most 256 KiB pending per direction per handler. Partial sends retain the unsent tail; backpressure stops reads instead of producing the old one-second payload-sendall timeout. EOF is propagated after that direction drains, and the opposite direction can finish. Every readiness loop and IO action checks original phase expiry. This does not promise transfer completion or a particular throughput. The short CONNECT handshake remains separately bounded; parent cleanup closes all known sockets. Terminal thread counts and actual process/group exit still require root review.

Eight focused synthetic tests pass, including multi-megabyte duplex payloads with many partial/blocked writes, exact bytes, buffer caps, initial CONNECT tail, EOF drainage, deadline interruption and closure-gate rejection. No live socket, actual clock/phase or process was used. Independent code_audit review and final package admission are pending.

## Root-only entry point and integration

Run `phase_proxy.py` using `--root-serve --phase-sha256 PIN --anchor-sha256 PIN --prior-closure-admission PATH --prior-closure-admission-sha256 PIN --output FRESH_ABSOLUTE_DIRECTORY` only after independent review. This process remains foreground in the original root tool. It emits ready schema `coder_sand_original_phase_recovery_route_proxy_v1` with the identical original `phase.metadata` fields, plus closure-admission SHA, connection/concurrency/buffer bounds and `payload_backpressure=nonblocking_partial_send`.

The operational recovery controller should validate that new ready schema and every original phase metadata field, and apply the uncredentialed loopback proxy only to its new SSH child. It should keep original failed capture/transfer tags intact, read saved worker products without rerunning scientific work, stream bounded chunks to fresh partial paths, and hash/length-check before exclusive publication. This route does not itself launch capture or transfer.

The prior closure admission must be root-issued JSON with schema `sand_original_phase_route_closure_admission_v1`, status `original_proxy_and_failed_local_transport_groups_closed`, and the exact phase/anchor/original-package pins. Required assertions are `original_proxy_tool_session=23611`, `original_proxy_pid=46711`, `original_proxy_pid_and_group_absent=true`, `failed_local_transport_groups_absent=[47926,48109]`, actual integer `original_proxy_tool_exit_code=130`, `scientific_workers_restarted=false`, and `new_or_restarted_clock_granted=false`. It additionally binds the three paths/hashes listed in `closure_evidence_review.json` under `original_proxy_tool_exit`, `original_proxy_terminal` and `local_native_closure`; each evidence file is rehashed before launch. The preparation has not manufactured this root admission.
