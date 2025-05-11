import os
from typing import List

import pandas as pd

import config
from preprocessing.translation import Translator
from util import logutil
from util.utils import format_long_types

logger = logutil.logger_run

NOISE = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"

NOISE_LIST = [
    "(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
    "(january|february|march|april|may|june|july|august|september|october|november|december)",
    "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    "\d{2}(:|.)\d{2}",
    "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
    "dear ((customer)|(user))",
    "dear",
    "(hello)|(hallo)|(hi )|(hi there)",
    "good morning",
    "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
    "thank you for contacting us",
    "thank you for your availability",
    "thank you for providing us this information",
    "thank you for contacting",
    "thank you for reaching us (back)?",
    "thank you for patience",
    "thank you for (your)? reply",
    "thank you for (your)? response",
    "thank you for (your)? cooperation",
    "thank you for providing us with more information",
    "thank you very kindly",
    "thank you( very much)?",
    "i would like to follow up on the case you raised on the date",
    "i will do my very best to assist you"
    "in order to give you the best solution",
    "could you please clarify your request with following information:"
    "in this matter",
    "we hope you(( are)|('re)) doing ((fine)|(well))",
    "i would like to follow up on the case you raised on",
    "we apologize for the inconvenience",
    "sent from my huawei (cell )?phone",
    "original message",
    "customer support team",
    "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
    "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
    "canada, australia, new zealand and other countries",
    "\d+",
    "[^0-9a-zA-Z]+",
    "(\s|^).(\s|$)",
]


def __load_raw_files() -> pd.DataFrame:
    df1 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/AppGallery.csv'), skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

    df2 = pd.read_csv(os.path.join(config.DATA_DIR, 'data/Purchasing.csv'), skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

    df = pd.concat([df1, df2])
    df[config.INTERACTION_CONTENT] = df[config.INTERACTION_CONTENT].values.astype('U')
    df[config.TICKET_SUMMARY] = df[config.TICKET_SUMMARY].values.astype('U')
    df[config.TICKET_SUMMARY] = df[config.TICKET_SUMMARY].fillna(' ')

    return df


def __remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    df[config.TICKET_SUMMARY] = df[config.TICKET_SUMMARY].str.lower().replace(NOISE, " ", regex=True).replace(
        r'\s+', ' ', regex=True).str.strip()
    df[config.INTERACTION_CONTENT] = df[config.INTERACTION_CONTENT].str.lower()

    for noise in NOISE_LIST:
        df[config.INTERACTION_CONTENT] = df[config.INTERACTION_CONTENT].replace(noise, " ", regex=True)
    df[config.INTERACTION_CONTENT] = df[config.INTERACTION_CONTENT].replace(r'\s+', ' ', regex=True).str.strip()

    return df


def preprocess_text_data(df: pd.DataFrame) -> pd.DataFrame:
    df[config.TICKET_SUMMARY] = df[config.TICKET_SUMMARY].fillna(' ')

    # translation
    df[config.TICKET_SUMMARY] = Translator().translate(df[config.TICKET_SUMMARY].tolist())

    # add `TEXT_COL` by joining `TICKET_SUMMARY` and `INTERACTION_CONTENT`
    df[config.TEXT_COL] = df[config.TICKET_SUMMARY] + ' ' + df[config.INTERACTION_CONTENT]

    # noise removal
    df = __remove_noise(df)

    return df


def __preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_text_data(df)

    # add `formatted_` type columns
    df[config.FORMATTED_TYPE_COLS] = df[config.TYPE_COLS].apply(format_long_types, axis=1)

    # add `CLASS_COL`
    df[config.CLASS_COL] = df[config.FORMATTED_TYPE_COLS[-1]].copy()

    # todo: remove
    df['y'] = df[config.CLASS_COL]

    # remove unlabelled data
    df = df.loc[(df['y'] != '') & df['y'].notna()]

    return df


def __load_preprocessed_df() -> pd.DataFrame:
    df = pd.read_csv(config.PREPROCESSED_FP)
    df[config.INTERACTION_CONTENT] = df[config.INTERACTION_CONTENT].values.astype('U')
    df[config.TICKET_SUMMARY] = df[config.TICKET_SUMMARY].values.astype('U')
    return df


def load_data(type1_list: List[str],
              preprocessed: bool) -> pd.DataFrame:
    if preprocessed:
        # when reading from already preprocessed file
        df = __load_preprocessed_df()
        if type1_list:
            df = df[df[config.GROUPED].str.strip().str.lower().isin(type1_list)]
        logger.info('Preprocessed DF is loaded.')
    else:
        # when reading from multiple raw data files
        df = __load_raw_files()
        if type1_list:
            df = df[df[config.GROUPED].str.strip().str.lower().isin(type1_list)]
        df = __preprocess_data(df)
        df.to_csv(config.PREPROCESSED_FP)
        logger.info('Data is preprocessed and saved.')

    logger.info(f"dataset size: {df.shape}")

    return df


"""
# todo: unused and remove
def de_duplication(data: pd.DataFrame):
    data["ic_deduplicated"] = ""

    cu_template = {
        "english":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"]
        ,
        "german":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?"]
        ,
        "french":
            ["L'équipe d'assistance à la clientèle d'Aspiegel\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?"]
        ,
        "spanish":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?"]
        ,
        "italian":
            ["Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?"]
        ,
        "portguese":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?"]
        ,
    }

    cu_pattern = ""
    for i in sum(list(cu_template.values()), []):
        cu_pattern = cu_pattern + f"({i})|"
    cu_pattern = cu_pattern[:-1]

    # -------- email split template

    pattern_1 = "(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = "(On.{30,60}wrote:)"
    pattern_3 = "(Re\s?:|RE\s?:)"
    pattern_4 = "(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = "(\s?\*\*\*\*\*\(PHONE\))*$"

    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"

    # -------- start processing ticket data

    tickets = data["Ticket id"].value_counts()

    for t in tickets.index:
        # print(t)
        df = data.loc[data['Ticket id'] == t,]

        # for one ticket content data
        ic_set = set([])
        ic_deduplicated = []
        for ic in df[config.INTERACTION_CONTENT]:

            # print(ic)

            ic_r = re.split(split_pattern, ic)
            # ic_r = sum(ic_r, [])

            ic_r = [i for i in ic_r if i is not None]

            # replace split patterns
            ic_r = [re.sub(split_pattern, "", i.strip()) for i in ic_r]

            # replace customer template
            ic_r = [re.sub(cu_pattern, "", i.strip()) for i in ic_r]

            ic_current = []
            for i in ic_r:
                if len(i) > 0:
                    # print(i)
                    if i not in ic_set:
                        ic_set.add(i)
                        i = i + "\n"
                        ic_current = ic_current + [i]

            # print(ic_current)
            ic_deduplicated = ic_deduplicated + [' '.join(ic_current)]
        data.loc[data["Ticket id"] == t, "ic_deduplicated"] = ic_deduplicated
    data.to_csv('out.csv')
    data[config.INTERACTION_CONTENT] = data['ic_deduplicated']
    data = data.drop(columns=['ic_deduplicated'])
    return data
"""
