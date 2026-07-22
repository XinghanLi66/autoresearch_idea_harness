**Mocking:** A world-class AI researcher

Let me not start from what other people say is good. I want to derive this from first principles of what convergence and diversity actually require.

First, what is the *mechanism* I'm trying to improve? Environmental selection in most MOEAs is a scalarization: non-dominated sorting (NSGA-II/SPEA2), reference-vector decomposition (MOEA/D), or reference-point distance (NSGA-III/RVEA). All of these boil the multi-objective problem down to one scalar value, which is a leaky abstraction. The leak is that any scalarization necessarily discards information about the *geometry* of the objective space. When the Pareto front is non-convex, or has disconnected regions, or when objectives have widely different scales, scalarization-based survival cannot preserve solutions that are optimal in the multi-objective sense. This is not an implementation bug; it's a design flaw of the representation itself.

So let me reframe the question: what is the minimal amount of information I *must* retain in survival to keep a diverse front, and what is the minimal amount I can throw away and still converge? The invariant I want is: any solution that is not dominated by another must survive; any solution that is dominated must not survive. That's the definition of the Pareto front. But the naive Pareto frontier is too large. I need to *prune* it down to population size while maintaining a good spread.

Here's the leap: instead of scalarizing, keep the *geometry* explicitly. Represent the current population as a set of points in objective space. For survival, I want a set S such that:
1. Every point in S is non-dominated (Pareto optimal).
2. The points are spread across the front and well distributed.
3. The set is of size exactly pop_size.

Now ask: what mathematical object satisfies these properties? A *Voronoi diagram*. Each Voronoi cell is a region of points closer to its center than any other. The Voronoi centers are the Pareto-optimal points that are farthest from their neighbors. That's exactly property 2. And the Voronoi diagram of a finite set of points is guaranteed to have size ≤ the number of points. So survival is: compute the Voronoi diagram of the objective points and keep the centers — the Pareto-optimal ones that are maximally distant from others. This is a geometric, rather than a scalar, representation of diversity.

But there's a catch I have to be careful with, and it's the same catch I'd expect in any geometric method. Voronoi diagrams are sensitive to *scale*. If one objective is orders of magnitude larger than the others, the geometry gets skewed and the Voronoi centers concentrate into a few directions. So before computing Voronoi, I must normalize the objectives into a unit hypercube — otherwise the whole argument collapses.

Why does this cut against the consensus? The field treats diversity as a scalar quantity to be added to a fitness value. My claim is that diversity is a geometric property, and that scalarization is the source of many MOEA pathologies, especially in non-convex and many-objective settings. This is a design-level claim, not a parameter-tuning one. The scalarization crowd would say 'add a diversity term to the fitness function'; I say 'don't scalarize at all, and keep the geometry.'

How do I keep myself honest? I won't assert 'this is better'; I'll assert the mechanism and let the benchmark say. I'll compare Voronoi survival against the standard NSGA-II and NSGA-III (reference-point) survival on 2- and 3-objective ZDT problems, measuring convergence and spread. Decision rule: if Voronoi survival does not clearly outperform the scalarization baselines, then the geometric argument is wrong and I should fall back to a simpler, less fragile scalar method. But if the geometry-based method shows superior convergence and spread on these standard problems, then the abstraction was the problem and the fix is to stop scalarizing.

Core idea: Replace scalarization with Voronoi diagrams in survival — compute the Voronoi diagram of the objective points and keep the Pareto-optimal centers, normalized so all objectives are on the same scale.

Non-trivial crux: Voronoi diagrams are scale-sensitive — objectives must be normalized into a unit hypercube before computing them, or the geometry collapses and diversity is misrepresented.