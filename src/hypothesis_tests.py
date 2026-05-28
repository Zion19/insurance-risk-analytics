# hypothesis_testing_module.py

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind


class HypothesisTesting:
    """
    Generic hypothesis testing framework for insurance risk analysis.

    Supports:
    - Flexible KPI parameter
    - Flexible grouping variable
    - Frequency and severity analysis
    - Profit / margin comparison
    - Gender risk comparison
    """

    def __init__(
        self,
        df: pd.DataFrame,
        kpi: str,
        group_column: str,
        group_a,
        group_b,
        alpha: float = 0.05
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe

        kpi : str
            KPI column to analyze
            Example:
            - TotalClaims
            - Margin
            - LossRatio

        group_column : str
            Column used for grouping
            Example:
            - Province
            - ZipCode
            - Gender

        group_a : any
            First comparison group

        group_b : any
            Second comparison group

        alpha : float
            Significance level
        """

        self.df = df.copy()
        self.kpi = kpi
        self.group_column = group_column
        self.group_a = group_a
        self.group_b = group_b
        self.alpha = alpha

        self.data_a = self.df[
            self.df[group_column] == group_a
        ].copy()

        self.data_b = self.df[
            self.df[group_column] == group_b
        ].copy()

    # ==================================================
    # CLAIM FREQUENCY ANALYSIS
    # ==================================================

    def frequency_test(self):
        """
        Tests whether claim frequency differs between groups.
        """

        self.data_a['HasEvent'] = (
            self.data_a[self.kpi] > 0
        ).astype(int)

        self.data_b['HasEvent'] = (
            self.data_b[self.kpi] > 0
        ).astype(int)

        subset = self.df[
            self.df[self.group_column].isin(
                [self.group_a, self.group_b]
            )
        ]

        contingency_table = pd.crosstab(
            subset[self.group_column],
            subset[self.kpi] > 0
        )

        chi2, p_value, dof, expected = chi2_contingency(
            contingency_table
        )

        return {
            "test": "Chi-Square Test",
            "group_column": self.group_column,
            "kpi": self.kpi,
            "group_a_frequency": self.data_a['HasEvent'].mean(),
            "group_b_frequency": self.data_b['HasEvent'].mean(),
            "p_value": p_value,
            "reject_null": p_value < self.alpha
        }

    # ==================================================
    # SEVERITY / MEAN DIFFERENCE TEST
    # ==================================================

    def severity_test(self):
        """
        Tests whether KPI severity/mean differs between groups.
        """

        values_a = self.data_a[
            self.data_a[self.kpi] > 0
        ][self.kpi].dropna()

        values_b = self.data_b[
            self.data_b[self.kpi] > 0
        ][self.kpi].dropna()

        # ==========================================
        # VALIDATION
        # ==========================================

        if len(values_a) == 0 or len(values_b) == 0:

            return {
                "test": "T-Test",
                "group_column": self.group_column,
                "kpi": self.kpi,
                "group_a_mean": values_a.mean(),
                "group_b_mean": values_b.mean(),
                "p_value": None,
                "reject_null": False,
                "message": (
                    "Severity test cannot be performed "
                    "because one group has no claims."
                )
            }

        # ==========================================
        # T-TEST
        # ==========================================

        result = ttest_ind(
            values_a,
            values_b,
            equal_var=False
        )

        p_value = result.pvalue

        return {
            "test": "T-Test",
            "group_column": self.group_column,
            "kpi": self.kpi,
            "group_a_mean": values_a.mean(),
            "group_b_mean": values_b.mean(),
            "p_value": p_value,
            "reject_null": p_value < self.alpha
        }

    # ==================================================
    # FULL HYPOTHESIS REPORT
    # ==================================================

    def run_hypothesis_test(self, null_hypothesis: str):
        """
        Runs complete statistical analysis.
        """

        print("=" * 60)
        print("HYPOTHESIS TESTING REPORT")
        print("=" * 60)

        print("\nNULL HYPOTHESIS")
        print("-" * 60)
        print(f"H₀: {null_hypothesis}")

        # ----------------------------------------------
        # Frequency Test
        # ----------------------------------------------

        freq_result = self.frequency_test()

        print("\n" + "=" * 60)
        print("FREQUENCY TEST")
        print("=" * 60)

        print(f"\nGroup Column: {self.group_column}")
        print(f"KPI: {self.kpi}")

        print(f"\n{self.group_a}: "
              f"{freq_result['group_a_frequency']:.4f}")

        print(f"{self.group_b}: "
              f"{freq_result['group_b_frequency']:.4f}")

        print(f"\nP-value: {freq_result['p_value']:.4f}")

        if freq_result['reject_null']:
            print("Reject H₀")
            print("There is a statistically significant difference.")
        else:
            print("Fail to Reject H₀")
            print("No statistically significant difference found.")
        # ----------------------------------------------
        # Severity / Mean Test
        # ----------------------------------------------

        severity_result = self.severity_test()

        print("\n" + "=" * 60)
        print("SEVERITY / MEAN DIFFERENCE TEST")
        print("=" * 60)

        print(f"\n{self.group_a} Mean: "
            f"{severity_result['group_a_mean']}")

        print(f"{self.group_b} Mean: "
            f"{severity_result['group_b_mean']}")

        # ==========================================
        # HANDLE INVALID TEST
        # ==========================================

        if severity_result["p_value"] is None:

            print("\nSeverity test could not be performed.")
            print(severity_result["message"])

        # ==========================================
        # VALID T-TEST RESULTS
        # ==========================================

        else:

            print(f"\nP-value: {severity_result['p_value']:.4f}")

            if severity_result['reject_null']:
                print("Reject H₀")
                print("Significant mean difference detected.")
            else:
                print("Fail to Reject H₀")
                print("No significant mean difference detected.")

        print("\n" + "=" * 60)
        print("END OF REPORT")
        print("=" * 60)