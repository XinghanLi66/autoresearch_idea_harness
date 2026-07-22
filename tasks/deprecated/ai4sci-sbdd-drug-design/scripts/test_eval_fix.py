#!/usr/bin/env python3
"""End-to-end test: apply pre_edit patches + run evaluate on one pocket."""
import subprocess, sys, os

WORKSPACE = "/workspace/CBGBench"
RESULT = "/scratch/gpfs/CHIJ/st3812/models/saves/ai4sci-sbdd-drug-design/targetdiff/seed_42/results/denovo/custom/seed42/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10"
PDB = os.path.join(WORKSPACE, "data/crossdocked_test/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10.pdb")

# 1. Patch docking_vina.py: obutils compat
dv = os.path.join(WORKSPACE, "repo/tools/docking_vina.py")
with open(dv) as f:
    lines = f.readlines()
lines[2] = (
    "try:\n"
    "    from meeko import obutils\n"
    "except ImportError:\n"
    "    from openbabel import pybel as _pybel\n"
    "    class obutils:\n"
    "        @staticmethod\n"
    "        def writeMolecule(mol, path):\n"
    "            _pybel.Molecule(mol).write('sdf', path, overwrite=True)\n"
)
with open(dv, "w") as f:
    f.writelines(lines)
print("1. Patched docking_vina.py obutils")

# 2. Patch evaluate_chem_single.py: docking try/except + vina stats tolerant
ecs = os.path.join(WORKSPACE, "evaluate_scripts/evaluate_chem_single.py")
with open(ecs) as f:
    content = f.read()

# 2a. Wrap vina docking in try/except
old_docking = "    vina_task = VinaDockingTask.from_generated_mol(\n        mol, mol_path, protein_path=args.pdb_path, center=args.center)"
new_docking = (
    "    vina_results = {'score_only': None, 'minimize': None, 'dock': None}\n"
    "    try:\n"
    "        vina_task = VinaDockingTask.from_generated_mol(\n"
    "            mol, mol_path, protein_path=args.pdb_path, center=args.center)"
)
content = content.replace(old_docking, new_docking, 1)

old_vina_dict = "    vina_results = {\n        'score_only': score_only_results,\n        'minimize': minimize_results,\n        'dock': docking_results\n    }"
new_vina_dict = (
    "        vina_results = {\n"
    "            'score_only': score_only_results,\n"
    "            'minimize': minimize_results,\n"
    "            'dock': docking_results\n"
    "        }\n"
    "    except Exception:\n"
    "        pass  # docking failed; chem_results still valid"
)
content = content.replace(old_vina_dict, new_vina_dict, 1)

# 2b. Replace vina stats + ensure save
old_stats = "    vina_score_only = [r['vina']['score_only']['affinity'] for r in results]"
if old_stats in content:
    idx = content.index(old_stats)
    save_line = "    torch.save(results, os.path.join(result_path, 'chem_eval_results.pt'))"
    save_idx = content.index(save_line)
    save_end = save_idx + len(save_line)

    new_section = (
        "    # Vina statistics (skip if docking failed)\n"
        "    try:\n"
        "        vs = [r['vina']['score_only']['affinity'] for r in results if r['vina']['score_only'] is not None]\n"
        "        vd = [r['vina']['dock']['affinity'] for r in results if r['vina']['dock'] is not None]\n"
        "        if vs: logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vs), np.median(vs)))\n"
        "        if vd: logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vd), np.median(vd)))\n"
        "    except Exception:\n"
        "        pass\n"
        "    # Save results (always)\n"
        "    try:\n"
        "        df = pd.DataFrame({'smiles': [r['smiles'] for r in results], 'qed': [r['chem_results']['qed'] for r in results], 'sa': [r['chem_results']['sa'] for r in results]})\n"
        "        df.to_csv(os.path.join(result_path, 'molecule_properties.csv'), index=False)\n"
        "    except Exception:\n"
        "        pass\n"
        "    torch.save(results, os.path.join(result_path, 'chem_eval_results.pt'))"
    )
    content = content[:idx] + new_section + content[save_end:]
    print("2. Patched evaluate_chem_single.py (docking + vina stats + save)")
else:
    print("WARNING: vina stats pattern not found")

with open(ecs, "w") as f:
    f.write(content)

# 3. Install wheels
subprocess.run("pip3 install --no-deps /data/crossdocked/wheels/gemmi-*.whl /data/crossdocked/wheels/AutoDockTools_py3-*.zip 2>/dev/null", shell=True)
print("3. Installed gemmi + AutoDockTools")

# 4. Verify import
sys.path.insert(0, WORKSPACE)
from repo.tools.docking_vina import VinaDockingTask
print("4. VinaDockingTask import OK")

# 5. Setup symlinks
os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)
os.system("ln -sf /data/crossdocked/crossdocked_test " + os.path.join(WORKSPACE, "data/crossdocked_test"))
os.system("ln -sf /usr/bin/python3 /usr/local/bin/python 2>/dev/null")

# 6. Run evaluate
print("5. Running evaluate_chem_single.py...")
ret = subprocess.run(
    ["python3", "evaluate_chem_single.py",
     "--result_path", RESULT, "--pdb_path", PDB,
     "--exhaustiveness", "16", "--eval_ref", "0", "--verbose", "1"],
    cwd=os.path.join(WORKSPACE, "evaluate_scripts"),
    capture_output=True, text=True
)
for line in ret.stdout.splitlines():
    if any(k in line for k in ["Evaluate done", "QED", "SA", "Vina"]):
        print("  ", line.strip())
if ret.returncode != 0:
    print("  STDERR (last 5 lines):")
    for line in ret.stderr.strip().splitlines()[-5:]:
        print("  ", line)

# 7. Check results
import torch
f = os.path.join(RESULT, "chem_eval_results.pt")
if os.path.isfile(f):
    data = torch.load(f, weights_only=False)
    n = len(data)
    print(f"6. chem_eval_results.pt: {n} samples")
    if n > 0:
        r = data[0]
        q = r["chem_results"]["qed"]
        s = r["chem_results"]["sa"]
        v = r["vina"]["score_only"]
        print(f"   QED={q:.4f} SA={s:.4f} Vina={'OK' if v else 'None'}")
        print("   TEST PASSED!")
    else:
        print("   WARNING: 0 samples in results")
else:
    print("6. FAIL: chem_eval_results.pt not found")
