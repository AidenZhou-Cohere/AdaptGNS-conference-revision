def run_training(args, helpers, support, process, device, runtime):
    torch = helpers.torch
    train, metadata, data_info = load_portable_dataset(args, helpers)
    config = {"schema": SCHEMA, "dataset": "Sand", "objective": args.objective,
              "seed": args.seed, "arm": args.arm, "updates": args.updates, "batch_size": 2, "history": 6,
              "initialization": "from scratch; paired seed across arms; empty Adam; no parent checkpoint",
              "graph_exposure": graph_exposure_config(args.arm),
              "architecture": {"width": 128, "message_passing_blocks": 10, "mlp_layers": 2},
              "noise_std": helpers.NOISE, "graph": {"radius": .015, "backend": "scipy_host", "cap": 128,
                                                       "self_candidates": True, "augmentation_probability": 0.0},
              "optimizer": {"name": "Adam", "initial_lr": 1e-4, "final_lr": 1e-5, "decay_updates": 100000,
                            "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0,
                            "foreach": False, "fused": False, "gradient_clipping": None},
              "checkpoint_every": args.checkpoint_every, "log_every": args.log_every,
              "data": data_info, "research_protocol_sha256": sha(args.protocol),
              "source_sha256": {**SOURCE_PINS, "train_sand_graph_support_cuda.py": sha(__file__)}, "runtime": runtime,
              "selection": "fixed final requested update; no validation/test selection"}
    config["portable_reproduction"] = args.portable_identity
    output = args.output_dir
    protected = (args.train_manifest.resolve().parent, (args.repo / "adaptive-gns").resolve())
    resolved_output = output.resolve()
    if any(resolved_output == path or path in resolved_output.parents or resolved_output in path.parents for path in protected):
        raise ValueError("Graph-support output must be separate from source/data trees")
    if list(output.rglob("*.tmp")):
        raise ValueError("Prior partial artifacts require review before any resume write")
    protocol = output / "protocol.json"
    if protocol.exists():
        if not args.resume or json.loads(protocol.read_text()) != config:
            raise ValueError("Existing output requires exact matching --resume configuration")
    elif args.resume or any(path.name != "run.lock" for path in output.iterdir()):
        raise ValueError("Resume requires protocol.json; fresh output must be empty")
    else:
        atomic_json(protocol, config)
    torch.manual_seed(args.seed)
    with torch.cuda.device(device):
        torch.cuda.manual_seed(args.seed)
    model = helpers.build_simulator(metadata, helpers.NOISE, helpers.NOISE, device,
        connectivity_radius=.015, nmessage_passing_steps=10, uncertainty_parameterization="variance",
        variance_floor=1e-6, detach_variance_features=True, radius_backend="scipy_host").to(device)
    model._training_config = {"loss": args.objective, "cuda_sand_graph_support_run": config}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False)
    completed, history, latest = 0, {"training": [], "graph_updates": [], "elapsed_seconds": 0.0}, None
    if args.resume:
        latest = read_pointer(output, config)  # Missing pointers require manual recovery; no silent restart.
        payload = torch.load(output / latest["path"], map_location="cpu", weights_only=True)
        completed, history = restore_payload(helpers, payload, model, optimizer, config, device)
        if completed != latest["completed_steps"]:
            raise ValueError("Pointer and checkpoint update counts disagree")
    stop = args.updates if args.stop_after is None else args.stop_after
    if stop < completed:
        raise ValueError("Requested stop precedes restored updates")
    started, prior_elapsed = time.perf_counter(), history["elapsed_seconds"]
    attempt_id = uuid.uuid4().hex
    attempt_dir = output / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    current_context = {"state": "before_first_update"}

    def status(state, error=None):
        record = {"schema": SCHEMA, "state": state, "attempt_id": attempt_id,
            "completed_steps": completed, "committed_steps": latest["completed_steps"] if latest else None,
            "requested_steps": args.updates, "objective": args.objective, "arm": args.arm, "seed": args.seed,
            "run_config_sha256": config_hash(config), "process": process, "updated_utc": utc_now(),
            "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
            "latest_checkpoint": latest, "last_training": history["training"][-1] if history["training"] else None,
            "error": error}
        atomic_json(attempt_dir / "status.json", record)
        atomic_json(output / "status.json", record)

    def checkpoint_now():
        nonlocal latest
        helpers.synchronize(device)
        history["elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
        payload = checkpoint_payload(helpers, model, optimizer, config, completed, history, device)
        path = output / f"checkpoint-{completed:09d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        if path.exists():
            raise ValueError("Refusing to overwrite an existing graph-support checkpoint")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        pointer = {"path": path.name, "sha256": sha(path), "completed_steps": completed,
                   "run_config_sha256": config_hash(config)}
        atomic_json(output / "latest.json", pointer)
        latest = pointer  # Report committed only after the atomic pointer publication succeeds.

    try:
        status("running")
        if latest is None:
            checkpoint_now()
        while completed < stop:
            helpers.synchronize(device)
            update_started = time.perf_counter()
            indices = helpers.sample_indices(args.seed, completed, len(train), 2)
            batch = helpers.unpack_batch([train[index] for index in indices])
            noise = helpers.host_noise(batch[0].shape, batch[1], args.seed, completed)
            frame_ids = [helpers.frame_identity(train, data_info["trajectory_ids"], index) for index in indices]
            current_context = {"completed_before": completed, "absolute_schedule_step": completed,
                "frame_ids": frame_ids, "noise_sha256": support.bridge.full.state_hash(noise.numpy())}
            rate = helpers.learning_rate(completed)
            for group in optimizer.param_groups:
                group["lr"] = rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred, head, target, graph_ledger = support.forward_batch(model, batch, noise, device, args.seed, completed, args.arm)
            current_context["examples"] = graph_ledger
            mask = (batch[1] != helpers.KINEMATIC).to(device)
            loss = guarded_update(helpers, model, optimizer, pred, head, target, mask, args.objective, completed + 1)
            helpers.synchronize(device)
            update_seconds = time.perf_counter() - update_started
            completed += 1
            history["graph_updates"].append({"completed_steps": completed, "absolute_schedule_step": completed - 1,
                "frame_ids": frame_ids, "noise_sha256": current_context["noise_sha256"], "examples": graph_ledger})
            if completed % args.log_every == 0 or completed in (1, stop):
                row = {"completed_steps": completed, "loss": float(loss.detach().cpu()), "lr": rate,
                       "frame_ids": frame_ids,
                       "particles": len(batch[0]), "guarded_update_seconds": update_seconds,
                       "elapsed_seconds": prior_elapsed + time.perf_counter() - started}
                history["training"].append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
                status("running")
            if completed % args.checkpoint_every == 0 or completed == stop:
                checkpoint_now()
        for relative, expected in config["source_sha256"].items():
            path = Path(__file__) if relative == "train_sand_graph_support_cuda.py" else args.repo / relative
            if sha(path) != expected:
                raise ValueError("Source changed during graph-support training: " + relative)
        for path, expected in ((args.train_manifest, data_info["manifest_sha256"]),
                               (args.metadata, data_info["metadata_sha256"]),
                               (args.protocol, config["research_protocol_sha256"])):
            if sha(path) != expected:
                raise ValueError("Input changed during graph-support training: " + str(path))
        reverify_portable_inputs(args, helpers, data_info)
        status("complete" if completed == args.updates else "planned_stop_incomplete")
        atomic_json(output / "history.json", history)
    except BaseException as error:
        atomic_json(attempt_dir / "unsuccessful_context.json", current_context)
        atomic_json(attempt_dir / "unsuccessful_history.json", history)
        try:
            with (attempt_dir / "unsuccessful_state.pt").open("xb") as stream:
                torch.save({"state_dict": helpers.cpu_tree(model.state_dict()),
                    "optimizer_state": helpers.cpu_tree(optimizer.state_dict()), "completed_steps": completed,
                    "error_type": type(error).__name__}, stream)
        except BaseException as preservation_error:
            atomic_json(attempt_dir / "state_preservation_error.json", {"error": str(preservation_error)})
        status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", f"{type(error).__name__}: {error}")
        raise
