"""An extensible fatty acid molecule builder."""

import mbuild as mb


class FattyAcid(mb.Compound):
    def __init__(self, chain_length=18, double_bonds=None, name="FA"):
        """An even more extensible fatty acid builder: Set chain lengths, number and position of the double bonds

        Parameters
        ----------
        chain_length : int
            Total number of carbons in the chain (including the carboxyl carbon).
        double_bonds : list of (position, geometry) tuples, or None
            position is counted from the carboxylic acid end,
            geometry is "cis" or "trans". None or [] gives a saturated chain.
            Examples: [(9, "cis")] -> oleic acid
                 [(9, "cis"), (12, "cis")] -> linoleic acid
        """
        super(FattyAcid, self).__init__()
        smiles = self._build_smiles(chain_length, double_bonds or [])
        mb.load(smiles, smiles=True, compound=self)
        self.name = name

    @staticmethod
    def _build_smiles(chain_length, double_bonds):
        bond_symbols = {"cis": "/C=C\\", "trans": "/C=C/"}
        double_bonds = sorted(double_bonds)
        chain = ["O=C(O)"]
        carbons_placed = 1
        for position, geometry in double_bonds:
            n_plain = position - 1 - carbons_placed
            chain.append("C" * n_plain)
            chain.append(bond_symbols[geometry])
            carbons_placed = position + 1

        chain.append("C" * (chain_length - carbons_placed))
        return "".join(chain)
