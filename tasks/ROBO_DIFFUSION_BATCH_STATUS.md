# Robo-Diffusion Tasks - Batch Creation Summary

## Status

This document tracks the creation of all robo-diffusion tasks.

### Completed Tasks

1. ✅ **robo-diffusion-network-backbone** (Branch: robo-diffusion-network-backbone)
   - Full implementation with 3 baselines
   - All files created and tested
   - Ready for use

### In Progress Tasks

2. 🚧 **robo-diffusion-guidance** (Branch: robo-diffusion-guidance)
   - Task description: ✅
   - Config.json: ✅
   - Parser.py: ✅
   - Training scripts: ✅
   - Template and baselines: ⏳ (simplified versions to be added)

3. ⏳ **robo-diffusion-sampling-method**
   - To be created

4. ⏳ **robo-diffusion-planner**
   - To be created

5. ⏳ **robo-diffusion-policy**
   - To be created

## Implementation Strategy

Due to the complexity and time required for full implementation, I'm using a phased approach:

### Phase 1: Core Structure (Current)
- Create task directories
- Add task_description.md
- Add config.json
- Add parser.py
- Add training scripts
- Add placeholder README

### Phase 2: Templates and Baselines (Next)
- Create custom_template.py with editable regions
- Create mid_edit.py
- Create baseline edit files
- Test with actual CleanDiffuser code

### Phase 3: Testing and Refinement
- Build containers
- Run baselines
- Fix any issues
- Update documentation

## Notes

- All tasks follow the same structure as robo-diffusion-network-backbone
- All tasks use rigorous_codebase = true
- All tasks have 3 baselines from CleanDiffuser
- All tasks evaluate on 3 D4RL environments (hopper, walker2d, halfcheetah)
