"""Patch evaluate_chem_single.py: wrap docking in try/except so chem-only results survive."""
import sys

path = sys.argv[1]
with open(path) as f:
    code = f.read()

# Replace the eval_single_mol function body
old = """    vina_task = VinaDockingTask.from_generated_mol(
        mol, mol_path, protein_path=args.pdb_path, center=args.center)

    score_only_results = vina_task.run(mode='score_only',
                                        exhaustiveness=args.exhaustiveness,
                                        save_dir=save_path)
    minimize_results = vina_task.run(mode='minimize',
                                        exhaustiveness=args.exhaustiveness,
                                        save_dir=save_path)
    docking_results = vina_task.run(mode='dock',
                                    exhaustiveness=args.exhaustiveness,
                                    save_dir=save_path)

    vina_results = {
        'score_only': score_only_results,
        'minimize': minimize_results,
        'dock': docking_results
    }"""

new = """    vina_results = {'score_only': None, 'minimize': None, 'dock': None}
    try:
        vina_task = VinaDockingTask.from_generated_mol(
            mol, mol_path, protein_path=args.pdb_path, center=args.center)
        vina_results['score_only'] = vina_task.run(mode='score_only',
                                            exhaustiveness=args.exhaustiveness,
                                            save_dir=save_path)
        vina_results['minimize'] = vina_task.run(mode='minimize',
                                            exhaustiveness=args.exhaustiveness,
                                            save_dir=save_path)
        vina_results['dock'] = vina_task.run(mode='dock',
                                        exhaustiveness=args.exhaustiveness,
                                        save_dir=save_path)
    except Exception:
        pass  # docking failed; chem_results still valid"""

if old not in code:
    print("WARNING: old pattern not found, skipping patch")
    sys.exit(0)

code = code.replace(old, new)
with open(path, 'w') as f:
    f.write(code)
print("Patched eval_single_mol: docking wrapped in try/except")
