===========
Biopolymers
===========

``mbuild.biopolymers`` loads a protonated protein PDB file, modifies it
covalently, and writes a file that a residue-template force-field tool
can ingest.

Why template matching
---------------------

Chemistry is never guessed. Every residue is matched against a template
from the wwPDB Chemical Component Dictionary (CCD) **by atom name**, and
the bonds, bond orders and formal charges come from that template. A
residue whose atoms no template variant explains raises an error naming
the residue and the fix, rather than producing a structure that is
quietly wrong.

That strictness is what the downstream tools need. Assigning force-field
parameters to a modified protein requires the element, the connectivity,
the bond orders and the formal charge of every atom. A reader that
infers bonds from interatomic distances supplies the first two and
guesses the rest, so the error surfaces later, during parameterization
or during the simulation.

Inter-residue bonds follow the same rule. A residue missing a whole
leaving fragment is a residue that is bonded to something: a missing
``H2`` means a peptide bond to the preceding residue, a missing
``OXT``/``HXT`` means one to the following residue, and a missing ``HG``
on a cysteine means a disulfide, which additionally requires a
``CONECT`` record between the two ``SG`` atoms.

The input must be fully protonated at the desired pH. mBuild adds no
hydrogens; prepare the file with PDBFixer or Reduce first.

Loading and inspecting
----------------------

.. code-block:: python

    from mbuild.biopolymers import Protein

    protein = Protein("1ubq_protonated.pdb")
    print(len(list(protein.residues())), protein.net_formal_charge)

    lysine = protein.get_residue(63, chain_id="A")
    print(lysine.formal_charge, lysine.atom_formal_charges)

The hierarchy is ``Protein -> Chain -> Residue -> particles``. Residues
keep their real PDB numbers, chain identifiers and insertion codes, so
``get_residue`` and ``get_atom`` address an atom the way the file does.

Some PDB files give an atom two or more alternate positions, marked in
the altLoc column. The loader takes the first conformer and warns. If
that is not the conformer you want, select it in your preparation tool
before loading.

Modifying a protein
-------------------

A fragment is written as SMILES with a ``*`` at the atom that forms the
bond. ``attach`` removes one leaving group from each side, a hydrogen
unless you name another atom, aligns the fragment along the two broken
bonds, and forms the new bond.

Prepare a charged site before bonding to it. ``attach`` puts the new
bond where a hydrogen was, so the formal charge of the anchor does not
change. A lysine side-chain amine is protonated at neutral pH and only
the neutral amine acylates, so deprotonate it first:

.. code-block:: python

    from mbuild.biopolymers import prepare_fragment

    fragment = prepare_fragment("*C(=O)CCCCCCC", "OC8")
    protein.deprotonate(63, "NZ", chain_id="A")
    protein.attach(fragment, resnum=63, atom_name="NZ", chain_id="A")

``deprotonate`` re-matches the residue to the CCD variant that describes
the result, so the template, the residue formal charge and the per-atom
formal charges all agree with the structure afterwards.

Point mutations
---------------

``mutate`` replaces the side chain of one residue and keeps its
backbone where the file put it. Name the residue and the new side
chain, as a CCD code:

.. code-block:: python

    protein.mutate(1500, "CYS", chain_id="A")

Everything beyond the bond from ``CA`` to the side chain leaves. The
new side chain is built from the CCD component, aligned along that old
bond direction, bonded to ``CA``, and moved into the residue, which
takes the new name and is re-matched to its definition. The side-chain
atom is found by its bond to ``CA``, not by its name, so a component
that calls it something other than ``CB`` works too. When the placed
side chain overlaps other atoms, it is relaxed together with the side
chains within 4 A of it; the backbone keeps the file's coordinates.
The same holds for a fragment placed by ``attach``. A
non-canonical amino acid the CCD holds, such as p-azido-L-phenylalanine
(``4II``), works the same way from a Protein built with
``download=True``, and the written file names the residue ``4II`` so
a residue library that knows the component reads it back:

.. code-block:: python

    protein = Protein("1fnf_clean.pdb", download=True)
    protein.mutate(1381, "4II", chain_id="A")

A side chain the CCD does not hold is written as a fragment, with a
``*`` at the atom that bonds to ``CA``. The marked atom becomes ``CB``:

.. code-block:: python

    azf = prepare_fragment("*Cc1ccc(N=[N+]=[N-])cc1", "AZF")
    protein.mutate(1381, azf, chain_id="A")

A CCD code gives the mutant the handedness of the component, read
from its ideal coordinates: ``"PHE"`` is L-phenylalanine and ``"DPN"``
is D-phenylalanine. A fragment keeps the handedness the residue had,
because the new ``CB`` goes where the old one was. ``stereo="D"`` or
``stereo="L"`` overrides either default by swapping the places of the
side chain and the alpha hydrogen. A glycine has no handedness and a
fragment makes it an L residue unless you ask for D. A CCD side chain arrives in the
component's default protonation state, the charged form for lysine,
arginine, aspartate and glutamate; ``deprotonate`` and ``protonate``
change it afterwards. Proline cannot be mutated to or from, because its
side chain bonds the backbone nitrogen.

Choosing which atom leaves
--------------------------

The hydrogens on one atom are chemically equivalent, so by default
``attach`` removes them in alphabetical order. A downstream residue
library instead describes the product by naming the atom that is
*absent*. When you are writing a file for such a library, name the
hydrogen it expects:

.. code-block:: python

    protein.attach(
        fragment,
        resnum=63,
        atom_name="NZ",
        chain_id="A",
        leaving_atom_names="HZ2",
        fragment_leaving_atom_names="H1",
    )

A leaving atom may also be a heavy atom bonded to the link atom. The
whole group on the far side of that bond leaves with it, and a residue
that this empties is dropped.

A fragment built from SMILES or read from a PDB file carries atom names
you did not choose. ``draw_fragment`` shows them: a 2D drawing with
every atom labelled, the bond sites in blue, any names you pass in red,
and a tint per residue when the fragment has several. Read the atom
that bonds and the atom that leaves off the picture before calling
``attach``:

.. code-block:: python

    from mbuild.biopolymers import draw_fragment, fragment_from_pdb

    glycan = fragment_from_pdb("glycan.pdb")
    draw_fragment(glycan, highlight="O1")   # shows C1 on residue 2, O1 and HO1 on ROH

Reactions given as a string
---------------------------

``attach`` replaces one bond on each side. A reaction that does more,
such as a click reaction that closes a ring or a Michael addition that
moves a hydrogen, is given as a reaction string: a name from
``mbuild.biopolymers.REACTIONS`` or an RDKit reaction SMARTS with the
protein side as the first template. The string is read as a rule and
never run on the protein. Atoms without a map number leave with the
group beyond them, product bonds between mapped atoms form with their
order, bonds that change order or vanish are edited, and mapped atoms
take the formal charges the product states. Only a hydrogen can be
created; a heavy atom the reactants do not supply is refused, because
nothing gives it coordinates.

.. code-block:: python

    from mbuild.biopolymers import REACTIONS

    print(REACTIONS["azide-alkyne triazole"])
    protein.attach(alkyne_dye, resnum=1381, atom_name="N3", chain_id="A",
                   reaction="azide-alkyne triazole")

The protein template must include the named atom, so the match is
local to the site. A reaction that forms several bonds between the
sides is placed by fitting the fragment to all of them and then relaxed
until every bond is at bond length. Each formed bond is recorded with
the reaction string, so ``bond_records()`` describes the product.

Shipped strings: ``hydrogen substitution`` (the default behaviour of
``attach``), ``amide coupling``, ``thiol-maleimide``,
``azide-alkyne triazole``, ``N-glycosylation`` and ``O-glycosylation``.
Where a template matches several heavy atoms of the fragment, as a
glycan's many hydroxyl carbons do, name the fragment atom.

A residue library declares one crosslink per residue, so a product that
joins the two sides by two bonds cannot be described as two residues.
Pass ``merge=True`` to put the fragment's atoms into the site residue
instead. The residue keeps its number, takes the fragment's charges and
drops its CCD template; rename it, and describe it to the library from
its own atoms and bonds:

.. code-block:: python

    site = protein.attach(alkyne_dye, resnum=1381, atom_name="N3", chain_id="A",
                          reaction="azide-alkyne triazole", merge=True)
    site.name = "TZ1"

Fragments from the CCD
----------------------

A fragment that is itself a CCD component, such as semaglutide's lipid
linker ``KUT``, is best built from that component. The atoms then carry
the names the CCD gives them, so a residue library that knows the
component reads the written file with no hand-written definition. The
link atom is recorded on the residue, so ``attach`` needs no
``fragment_atom_name``:

.. code-block:: python

    from mbuild.biopolymers import fragment_from_ccd

    linker = fragment_from_ccd("KUT", link_atom="C33")
    protein.attach(linker, resnum=20, atom_name="NZ", chain_id="A")

Attaching a glycan from a PDB file
----------------------------------

A glycan from a glycan builder has residue names that no template
library knows, and a force field such as GLYCAM06 assigns its
parameters by exactly those names. ``fragment_from_pdb`` reads such a
file with its residues intact. It takes the bonds from the ``CONECT``
records, which glycan builders write, and perceives the bond orders
from the connectivity and coordinates with RDKit.

The builder ends the glycan in a hydroxyl residue (``ROH``) on the
anomeric carbon. The glycosidic bond forms where that hydroxyl was, so
name its oxygen as the fragment's leaving atom. Asparagine's amide
nitrogen is neutral, so it needs no ``deprotonate`` first; ``attach``
removes the one hydrogen it replaces:

.. code-block:: python

    from mbuild.biopolymers import fragment_from_pdb

    glycan = fragment_from_pdb("glycan.pdb")
    protein.attach(
        glycan,
        fragment_atom_name="C1",
        fragment_resnum=2,
        resnum=60,
        atom_name="ND2",
        chain_id="A",
        leaving_atom_names="HD22",
        fragment_leaving_atom_names="O1",
    )

The ``ROH`` residue is gone from the product, and the sugar residues
keep their names, renumbered to continue the chain. The same call
O-links a glycan to a serine or threonine: name ``OG`` or ``OG1`` as
the atom and its hydroxyl hydrogen as the leaving atom.

Residue names are force-field vocabulary, so ``attach`` leaves the
protein residue's name alone. A force field that has parameters for
the glycosylated residue under its own name, as GLYCAM06 does with
``NLN``, ``OLS`` and ``OLT``, gets that name in one line, and
``save_pdb`` and ``bond_records`` both report it:

.. code-block:: python

    protein.get_residue(60, chain_id="A").name = "NLN"

Exporting
---------

``save_pdb`` writes residue-ordered ATOM and HETATM records with real
PDB residue numbers and chain identifiers, a TER after each chain, and
CONECT records for exactly the bonds that residue adjacency cannot
imply. Peptide bonds stay implied, because a strict residue-template
reader fails on a CONECT its definitions cannot explain.

``bond_records()`` returns one plain dict per inter-residue bond that
adjacency does not imply: the two residues, the two bonded atoms, the
leaving atoms and the bond order. Together with the PDB file, that is
everything a downstream tool needs to describe the modification.

``visualize`` returns an NGLView widget built from the same PDB text
that ``save_pdb`` writes: the protein as a cartoon, attached residues as
licorice. Return it from a notebook cell to display it.

``to_rdkit`` returns a sanitized ``Mol`` carrying the formal charges and
bond orders the loader derived, a conformer built from the particle
positions, and PDB residue information per atom. It raises on a bond
without an order, so it doubles as a check that the chemistry survived.

API
---

.. autoclass:: mbuild.biopolymers.Protein
    :members:

.. autoclass:: mbuild.biopolymers.Chain
    :members:

.. autoclass:: mbuild.biopolymers.Residue
    :members:

.. autoclass:: mbuild.biopolymers.CCDLibrary
    :members:

.. autofunction:: mbuild.biopolymers.prepare_fragment

.. autofunction:: mbuild.biopolymers.fragment_from_smiles

.. autofunction:: mbuild.biopolymers.fragment_from_pdb

.. autofunction:: mbuild.biopolymers.fragment_from_ccd

.. autofunction:: mbuild.biopolymers.draw_fragment
