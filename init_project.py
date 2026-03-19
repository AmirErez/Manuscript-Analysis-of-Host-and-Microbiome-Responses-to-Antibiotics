import os

PRIVATE_SUBDIRS = [
    "Private",
    "Private/clusters_properties",
    "Private/CompoResGenes",
    "Private/CompoResultsPlots",
    "Private/CompoResVerification",
    "Private/DIABLO_Analysis_Outputs",
    "Private/DIABLO_Analysis_Outputs_pairs",
    "Private/random_tightness",
    "Private/analysis",
]

def create_private_dirs():
    for folder in PRIVATE_SUBDIRS:
        path = os.path.join(os.getcwd(), folder)
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created folder: {path}")

if __name__ == "__main__":
    create_private_dirs()
