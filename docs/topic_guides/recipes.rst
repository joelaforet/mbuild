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
