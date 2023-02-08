WANDB_PROJECT = "big-cat-classification"
ENTITY = None # set this to team name if working in a team
# BDD_CLASSES = {i:c for i,c in enumerate(['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle'])}
LIONS_CLASSES = {i:c for i,c in enumerate(['Lion', 'Lioness', 'Lion Cub'])}
BIG_CAT_CLASSES = {i:c for i,c in enumerate(['Black panther', 'Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Snow Leopard', 'Tiger'])}
RAW_DATA_AT = 'bcc_bigcats'
CLEAN_DATA_AT = 'bcc_bigcats_clean'
PROCESSED_DATA_AT = 'bcc_bigcats_split'