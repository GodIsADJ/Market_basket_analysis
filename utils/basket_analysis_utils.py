import pandas as pd
import numpy as np
import csv

# Create encoder function
def encode_units(x) -> int:
    """Create encoder function, if quantity > 0 we return 1, else we return 0.

    Args:
        x (int): the quantity we wants to encode

    Returns:
        int: 0 or 1
    """
    if x <= 0:
        return 0
    if x >= 1:
        return 1


def remove_duplicate(df) -> pd.DataFrame:
    """Remove identical orders made by the same customer

    Args:
        df (pd.DataFrame): The data we wants to clean

    Returns:
        pd.DataFrame: The new dataframe
    """
    order_id_list = df.id_order.unique()
    lib = [list(df[df["id_order"] == o].product_id) for o in order_id_list]
    lib2 = [list(df[df["id_order"] == o].id_customer) for o in order_id_list]
    lib3 = [list(df[df["id_order"] == o].id_order) for o in order_id_list]
    test = pd.DataFrame(columns=["customer", "products"])
    for i in range(len(lib)):
        prod = " ".join(str(elem) for elem in lib[i])
        test = test.append(
            {"customer": lib2[i][0], "products": prod, "order": lib3[i][0]},
            ignore_index=True,
        )
    not_duplicated = test.drop_duplicates(subset=["customer", "products"])
    lib4 = []
    for i in range(len(lib3)):
        lib4.append(lib3[i][0])
    duplicate_order = []
    for o in lib4:
        if o not in not_duplicated["order"].unique():
            duplicate_order.append(o)
    df = df.drop(duplicate_order)
    return df


def save_frequency_to_csv(frequent_itemsets) -> None:
    """Save list of items with frequency in a csv file

    Args:
        frequent_itemsets (pd.DataFrame): the list of products and the frequency
    """
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
    with open("data/frequent_itemset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["products", "frequency"])
        for i in frequent_itemsets.index:
            products = tuple(frequent_itemsets["itemsets"][i])
            frequency = "{0:.2f}%".format(frequent_itemsets["support"][i] * 100)
            writer.writerow([products, frequency])


def zhang(antecedent, consequent) -> float:
    """Compute Zhang's metric

    Args:
        antecedent (pd.Series): 0 (product not include in the order) or 1 (product include in the order) for each order
        consequent (pd.Series): 0 (product not include in the order) or 1 (product include in the order) for each order

    Returns:
        float: the zhang's metric
    """
    # Compute the support of each item
    supportA = antecedent.mean()
    supportC = consequent.mean()

    # Compute the support of both items
    supportAC = np.logical_and(antecedent, consequent).mean()

    # Complete the expressions for the numerator and denominator
    numerator = supportAC - supportA * supportC
    denominator = max(supportAC * (1 - supportA), supportA * (supportC - supportAC))

    # Return Zhang's metric
    return numerator / denominator
