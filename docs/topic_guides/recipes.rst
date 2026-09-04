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

Some PDB files give an atom two or more alternate positions, marked A,
B, and so on in the altLoc column. The loader takes the first conformer
in the file: it keeps the records with no altLoc mark and the records
that carry the first mark, usually A. It drops the other positions and
lists the affected residues in a warning. The first conformer is not
always the one you want to model. Choose the conformer before you load
the file: select it in your preparation tool, or with PyMOL or a text
editor, and remove the other altLoc records. Then the loader has one
position per atom and gives no warning.

Prepare a charged site before you bond to it. ``attach`` removes one
hydrogen from the anchor atom and puts the new bond in its place, so
the formal charge of that atom does not change. A lysine side-chain
amine is protonated at neutral pH, and only the neutral amine acylates,
so call ``Protein.deprotonate`` first:

.. code-block:: python

    protein.deprotonate(63, "NZ", chain_id="A")
    protein.attach(fragment, resnum=63, atom_name="NZ", chain_id="A")

A written modified protein does not load again in mBuild. The attached
fragment has no CCD entry, and mBuild declares no modification bond in
the file, so the loader cannot match the fragment residue. Hand the PDB
file and ``bond_records()`` to a downstream tool, which builds the
residue definition for the modification.

.. warning::

    ``Compound.save`` writes a wrong ``.gro`` file for a protein, and it
    raises no error. It sends ``.gro`` to the generic GMSO converter,
    which numbers residues by counting each residue name, so the file
    holds one residue named ``Chain`` for the whole protein. Write the
    packed system with ``mb.biopolymers.save`` instead. It gives every
    residue its own name and number.

.. autoclass:: mbuild.biopolymers.protein.Protein
    :members:

.. autoclass:: mbuild.biopolymers.protein.Chain
    :members:

.. autoclass:: mbuild.biopolymers.protein.Residue
    :members:

.. autoclass:: mbuild.biopolymers.ccd.CCDLibrary
    :members:

.. autofunction:: mbuild.biopolymers.fragments.prepare_fragment

.. autofunction:: mbuild.biopolymers.fragments.fragment_from_smiles

Solvating a protein
^^^^^^^^^^^^^^^^^^^

Modify and relax the protein before you solvate it. PACKMOL keeps the
solute rigid, so a clash between a fragment and the protein at packing
time stays in the packed system.

``mb.solvate`` returns a Compound that holds the protein as its first
child. That child is still the ``Protein`` object, so you can still call
``save_pdb``, ``bond_records``, ``get_residue`` and the other
``Protein`` methods on it.

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
    box = mb.Box([7.0, 7.0, 7.0])
    system = mb.solvate(protein, water, 10000, box)

    solute = system.children[0]  # still a Protein
    solute.save_pdb("solute.pdb")
    print(solute.bond_records())

    mb.biopolymers.save(system, "system.gro", box=box)

.. warning::

    ``Compound.save`` writes a wrong ``.gro`` file for a packed protein,
    and it raises no error. It sends ``.gro`` to the generic GMSO
    converter, which numbers residues by counting each residue name, so
    the file holds one residue named ``Chain`` for the whole protein.
    Write the packed system with ``mb.biopolymers.save`` instead. It
    gives every residue its own name and number.

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
- ``mb.solvate`` and ``mb.fill_box`` do not set a box on the packed
  system. Pass ``box=`` to ``mb.biopolymers.save`` for a ``.gro`` or a
  ``.top`` file, as the snippet above does. Without it the writer takes
  the bounding box of the compound, which is smaller than the packing
  box, and ``save`` logs a warning.

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
