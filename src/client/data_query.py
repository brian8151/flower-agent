from random import random


class DataQuery:
    """ Class for Data Query service """
    def __init__(self):
        # Simulate a simple in-memory "database"
        self.data_storage = []
        self.results_storage = []

    def save_prediction_results(self, results):
        # Add "is_correct" field to each result
        for result in results:
            result["is_correct"] = random.choice(["Y", "N"])
        self.results_storage.extend(results)

    def get_prediction_results(self):
        # Retrieve saved prediction results
        return self.results_storage[:]

    def get_payment_data(self):
        # Each tuple corresponds to a row from the payment data table shown in your image
        return [
                ("GB24BIBA11216084705465", "Robert Miller", "GB37WMCW65007687495948", "Aaron Jones", 20604, -390.13,
                 0.76, 1),
                ("GB12GOPX26544950239156", "Jenna Webb", "GB62NPGR81284599326073", "Sharon Torres", 21303, -1688.11,
                 0.88, 5),
                ("GB33WMXE47131809299427", "Kelly Parker", "GB81UQRW93838903091482", "Haley Heath", 31453, 695.59, 0.81,
                 7),
                ("GB27YDTI53538336797439", "Robyn Lewis", "GB45WGXW08317592542873", "Michelle Carter", 84606, -2302.61,
                 0.79, 0),
                ("GB48SPLB20653006099340", "Nicole Crawford", "GB72AKVF76618348720752", "Jennifer Powell", 80281,
                 2697.89, 0.92, 1),
                ("GB58BLNY39992294241734", "Bonnie Jackson", "GB15CPAX55573720709512", "Kendra Lewis", 12894, -2680.52,
                 0.999, 9),
                ("GB21FUBW13686310302802", "Ruben Thomas", "GB56AIKJ90738271705213", "Michele Edwards", 30278, 565.45,
                 0.89, 2),
                ("GB90IFRJ06238827498287", "Gary Davis", "GB94JLDB43040679008325", "Belinda Warren", 80323, -132.23,
                 0.96, 1),
                ("GB90OLQR90227240221468", "Paul Webb", "GB34QUXS65995631490009", "Joanna Mendez", 75689, 137.97, 0.78,
                 3),
                ("GB77MHEI15024037568776", "Heather Reyes", "GB92UMXN45867449347917", "Lori Smith", 88564, -3933.18,
                 0.88, 10),
                ("GB54YFER04299837154465", "William James", "GB33CYPX61198869692216", "Molly Wu", 50081, -1501.62, 0.87,
                 10),
                ("GB98FZDB99467780296684", "Kathleen Spence", "GB48PGRH89021454266244", "Anne Nguyen", 29561, 407.77,
                 0.996, 10),
                ("GB96NTOH37673553164326", "Alyssa Gomez", "GB58OXIJ00940406612347", "Stephen Kim", 16321, -2495.47,
                 0.987, 10),
                ("GB82REBK56708895580142", "James Molina", "GB14MRVJ46566273656217", "Jessica Smith", 59563, 4164.26,
                 0.927, 7),
                ("GB98JCMV25693856190196", "Christopher Black", "GB29OQOS46134606231640", "Mrs. Stephanie Thomas", 51476,
                1187.14, 0.965, 5),
                ("GB27MGSL64075577504675", "Marcus Jenkins DDS", "GB25EMRW90554119140896", "Joseph Clark", 91007,
                 -4717.40, 0.884, 8),
                ("GB13MLQW49837406568421", "Diana Gregor", "GB37MLFD05387498640372", "Natalie Brown", 45000, 1200.00,
                 0.95, 2),
                ("GB82REBK56708895580142", "James Molina", "GB14MRVJ46566273656217", "Jessica Smith", 59563, 4164.26,
                 0.927, 7),
                ("GB28UIBN40232912325984", "Roger Brown", "GB41HKXC73814198993753", "Marcus Brown", 71476,
                    1179.14, 0.925,9),
                ("GB68PHEJ90508421728212", "Anna Edwards", "GB51WFXE87172808920069", "Juan Logan", 81007,
                 -4762.40, 0.814, 2)
        ]