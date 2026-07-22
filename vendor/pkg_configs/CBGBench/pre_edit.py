"""Pre-edit operations for CBGBench package.
1. Register 'custom' model in repo/models/__init__.py
2. Inject TRAIN_METRICS into train.py (print loss/lr every report_freq iterations)
"""

OPS = [
    {
        "op": "replace",
        "file": "CBGBench/repo/models/__init__.py",
        "start_line": 1,
        "end_line": 3,
        "content": (
            "from .diffusion.diffusion import *\n"
            "from .autoregression.autoregression import *\n"
            "from ._base import *\n"
            "\n"
            "# Register custom model\n"
            "try:\n"
            "    from repo.models.custom_sbdd import CustomSBDD\n"
            "except ImportError:\n"
            "    pass\n"
        ),
    },
    {
        "op": "replace",
        "file": "CBGBench/train.py",
        "start_line": 196,
        "end_line": 205,
        "content": (
            "        if it % config.train.report_freq == 0:\n"
            "            # Logging\n"
            "            scalar_dict = {}\n"
            "            scalar_dict.update({\n"
            "                'grad': orig_grad_norm,\n"
            "                'lr': optimizer.param_groups[0]['lr'],\n"
            "                'time_forward': (time_forward_end - time_start) / 1000,\n"
            "                'time_backward': (time_backward_end - time_forward_end) / 1000,\n"
            "            })\n"
            "            log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger, writer=writer)\n"
            "            # TRAIN_METRICS injection\n"
            "            loss_parts = ' '.join(f'{k}={v.item():.6f}' for k, v in loss_dict.items())\n"
            "            print(f'TRAIN_METRICS iter={it} loss={loss.item():.6f} {loss_parts} lr={optimizer.param_groups[0][\"lr\"]:.2e}', flush=True)\n"
        ),
    },
    # Fix get_new_log_dir bug: tag replaces fn instead of appending.
    # Keep the buggy behavior (fn=tag) since our scripts now match it,
    # but this comment documents it for future reference.

    # --- Evaluation compatibility fixes for meeko 0.7.1 ---
    # meeko 0.7.1 removed obutils; provide pybel-based fallback
    {
        "op": "replace",
        "file": "CBGBench/repo/tools/docking_vina.py",
        "start_line": 3,
        "end_line": 3,
        "content": (
            "try:\n"
            "    from meeko import obutils\n"
            "except ImportError:\n"
            "    from openbabel import pybel as _pybel\n"
            "    class obutils:\n"
            "        @staticmethod\n"
            "        def writeMolecule(mol, path):\n"
            "            _pybel.Molecule(mol).write('sdf', path, overwrite=True)\n"
        ),
    },
    # Wrap Vina docking in eval_single_mol with try/except so chem results
    # survive even when meeko/vina docking fails (API incompatibility)
    {
        "op": "replace",
        "file": "CBGBench/evaluate_scripts/evaluate_chem_single.py",
        "start_line": 44,
        "end_line": 61,
        "content": (
            "    vina_results = {'score_only': None, 'minimize': None, 'dock': None}\n"
            "    try:\n"
            "        vina_task = VinaDockingTask.from_generated_mol(\n"
            "            mol, mol_path, protein_path=args.pdb_path, center=args.center)\n"
            "        vina_results['score_only'] = vina_task.run(mode='score_only',\n"
            "                                            exhaustiveness=args.exhaustiveness,\n"
            "                                            save_dir=save_path)\n"
            "        vina_results['minimize'] = vina_task.run(mode='minimize',\n"
            "                                            exhaustiveness=args.exhaustiveness,\n"
            "                                            save_dir=save_path)\n"
            "        vina_results['dock'] = vina_task.run(mode='dock',\n"
            "                                        exhaustiveness=args.exhaustiveness,\n"
            "                                        save_dir=save_path)\n"
            "    except Exception:\n"
            "        pass  # docking failed; chem_results still valid\n"
        ),
    },
    # Replace vina stats + DataFrame + reference eval (lines 120-168, entire tail)
    # Combined into one op to avoid line-number shift bugs between sequential ops
    {
        "op": "replace",
        "file": "CBGBench/evaluate_scripts/evaluate_chem_single.py",
        "start_line": 120,
        "end_line": 168,
        "content": (
            "    # Vina statistics (skip if docking failed)\n"
            "    try:\n"
            "        vina_score_only = [r['vina']['score_only']['affinity'] for r in results if r['vina']['score_only'] is not None]\n"
            "        vina_min = [r['vina']['minimize']['affinity'] for r in results if r['vina']['minimize'] is not None]\n"
            "        if vina_score_only:\n"
            "            logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))\n"
            "        if vina_min:\n"
            "            logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))\n"
            "        vina_dock = [r['vina']['dock']['affinity'] for r in results if r['vina']['dock'] is not None]\n"
            "        if vina_dock:\n"
            "            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))\n"
            "    except Exception:\n"
            "        pass\n"
            "\n"
            "    # Always save evaluation results first\n"
            "    torch.save(results, os.path.join(result_path, 'chem_eval_results.pt'))\n"
            "\n"
            "    # Build DataFrame with available data\n"
            "    try:\n"
            "        result_filter = [r for r in results if r['vina']['dock'] is not None and r['vina']['dock']['affinity'] < 0]\n"
            "        if result_filter:\n"
            "            vina_dock = [r['vina']['dock']['affinity'] for r in result_filter]\n"
            "            vina_dock_idx = np.argsort(vina_dock)\n"
            "            file_names = [result_filter[i]['ligand_filename'] for i in vina_dock_idx]\n"
            "            chem_results_list = [result_filter[i]['chem_results'] for i in vina_dock_idx]\n"
            "            df = pd.DataFrame({\n"
            "                'file_names': [os.path.split(f)[1] for f in file_names],\n"
            "                'smiles': [result_filter[i]['smiles'] for i in vina_dock_idx],\n"
            "                'vina_dock_result': vina_dock,\n"
            "                'qed': [c['qed'] for c in chem_results_list],\n"
            "                'sa': [c['sa'] for c in chem_results_list],\n"
            "            })\n"
            "        else:\n"
            "            df = pd.DataFrame({\n"
            "                'file_names': [r['ligand_filename'] for r in results],\n"
            "                'smiles': [r['smiles'] for r in results],\n"
            "                'qed': [r['chem_results']['qed'] for r in results],\n"
            "                'sa': [r['chem_results']['sa'] for r in results],\n"
            "            })\n"
            "        df.to_csv(os.path.join(result_path, 'molecule_properties.csv'), index=False)\n"
            "    except Exception:\n"
            "        pass\n"
            "\n"
            "    if args.eval_ref:\n"
            "        try:\n"
            "            ref_mol_path = os.path.join(os.path.dirname(args.pdb_path), '_'.join(os.path.basename(args.pdb_path).split('_')[:-1]) + '.sdf')\n"
            "            ref_result = eval_single_mol(ref_mol_path, dock_result_path)\n"
            "            torch.save(ref_result, os.path.join(result_path, 'chem_reference_results.pt'))\n"
            "            logger.info('Reference ligand evaluation done!')\n"
            "\n"
            "            ref_row = {'file_names': 'reference', 'smiles': ref_result['smiles'],\n"
            "                       'qed': ref_result['chem_results']['qed'], 'sa': ref_result['chem_results']['sa'],\n"
            "                       'logp': ref_result['chem_results']['logp'], 'lipinski': ref_result['chem_results']['lipinski']}\n"
            "            if ref_result['vina']['dock'] is not None:\n"
            "                ref_row['vina_dock_result'] = ref_result['vina']['dock']['affinity']\n"
            "            if ref_result['vina']['minimize'] is not None:\n"
            "                ref_row['vina_min_result'] = ref_result['vina']['minimize']['affinity']\n"
            "            if ref_result['vina']['score_only'] is not None:\n"
            "                ref_row['vina_score_result'] = ref_result['vina']['score_only']['affinity']\n"
            "            df = pd.read_csv(os.path.join(result_path, 'molecule_properties.csv'))\n"
            "            df = pd.concat([df, pd.DataFrame([ref_row])], ignore_index=True)\n"
            "            df.to_csv(os.path.join(result_path, 'molecule_properties.csv'), index=False)\n"
            "        except Exception:\n"
            "            pass  # reference evaluation failed; main results already saved\n"
        ),
    },
]
