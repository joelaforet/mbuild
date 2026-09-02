Recipes
=======

Monolayer
---------

.. autoclass:: mbuild.lib.recipes.monolayer.Monolayer
    :members:

Polymer
-------

.. autoclass:: mbuild.lib.recipes.polymer.Polymer
    :members:

Biopolymers
-----------

The biopolymers recipe loads a protonated protein PDB file and assigns
template chemistry from the Chemical Component Dictionary (CCD). It can
covalently modify the protein with fragment compounds. It writes a PDB
file with bond records for downstream parameterization. Worked examples
live in the demos repository:
https://github.com/joelaforet/mbuild_protein_demos

.. autoclass:: mbuild.biopolymers.protein.Protein
    :members:

.. autoclass:: mbuild.biopolymers.protein.Chain
    :members:

.. autoclass:: mbuild.biopolymers.protein.Residue
    :members:

.. autoclass:: mbuild.biopolymers.ccd.CCDLibrary
    :members:

.. autofunction:: mbuild.biopolymers.fragments.prepare_fragment

.. autofunction:: mbuild.biopolymers.fragments.fragment_from_sdf

.. autofunction:: mbuild.biopolymers.fragments.fragment_from_smiles

Solvating a protein
^^^^^^^^^^^^^^^^^^^

Attach every fragment and relax it before the protein is packed:
PACKMOL holds the solute rigid, so a clash that is present at packing
time stays in the packed system. ``mb.solvate`` keeps the recipe's
hierarchy, so the solute child of the packed system is still a
``Protein`` and still answers ``save_pdb``, ``bond_records`` and the
residue accessors.

.. code-block:: python

    import mbuild as mb
    from mbuild.biopolymers import Protein, prepare_fragment
    from mbuild.lib.molecules.water import WaterSPC

    protein = Protein("1ubq_protonated.pdb")
    fragment = prepare_fragment("*C(=O)CCCCCCC", "OCT")
    protein.attach(fragment, resnum=63, atom_name="NZ", chain_id="A")
    protein.relax_fragments()

    water = WaterSPC()
    water.name = "SOL"
    system = mb.solvate(protein, water, 10000, mb.Box([7.0, 7.0, 7.0]))

    solute = system.children[0]  # still a Protein
    solute.save_pdb("solute.pdb")
    print(solute.bond_records())

    mb.biopolymers.save(system, "system.gro")

Write the packed system with ``mb.biopolymers.save``, not with
``Compound.save``. ``Compound.save`` sends ``.gro`` to the module-level
GMSO converter, which numbers residues by counting each residue name,
so the file holds one residue named ``Chain`` for the whole protein.
``mb.biopolymers.save`` gives every residue its own name and number.

Three further points:

- ``mb.fill_box`` gives each copy a random rotation. Pass
  ``fix_orientation=True`` to keep the loaded orientation. ``mb.solvate``
  places the solute with PACKMOL's ``fixed`` restraint, so the solute
  never rotates.
- A ``.pdb`` file of the packed system comes from the generic ParmEd
  writer, which needs the residue names:
  ``system.save("system.pdb", residues=[...])``, with the solvent name
  (``"SOL"`` above) in the list. Use ``solute.save_pdb`` for the
  protein alone.
- A ``.top`` file needs a force field that types the protein, and it
  needs a change in GMSO's top writer, which puts the moleculetype name
  in the residue column.

Cost: packing 8ciq in 10000 waters (30549 atoms) took about 10 s and
the ``.gro`` write about 5 s on one desktop core. Both grow linearly
with the atom count.

Tiled Compound
--------------

.. autoclass:: mbuild.lib.recipes.tiled_compound.TiledCompound
    :members:

Silica Interface
----------------

.. autoclass:: mbuild.lib.recipes.silica_interface.SilicaInterface
    :members:

Packing
-------
.. automodule:: mbuild.packing
    :members:

Pattern
-------
.. automodule:: mbuild.pattern
    :members:
