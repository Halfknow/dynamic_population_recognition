import numpy as np

features = {
    'cat__SAN_YUE_YI_YUAN_0': 1,
    'cat__YI_NIAN_YI_YUAN_0': 0,
    'cat__YI_NIAN_YI_YUAN_1': 1,
    'cat__DANG_DI_JIAO_SHI_0': 1,
    'cat__DANG_DI_JIAO_SHI_1': 0,
    'cat__DANG_DI_XUE_SHENG_0': 1,
    'cat__DANG_DI_XUE_SHENG_1': 0,
    'cat__DANG_DI_ZAI_ZHI_0': 0,
    'cat__DANG_DI_ZAI_ZHI_1': 1,
    'cat__GONG_JI_JIN_0': 1,
    'cat__BU_DONG_CHAN_SHU_LIANG_0': 1,
    'cat__GONG_ZU_FANG_0': 1,
    'cat__GONG_ZU_FANG_1': 0,
    'cat__JIN_QI_JIAO_YI_0': 1,
    'cat__SHE_HUI_JIU_ZHU_0': 1,
    'cat__SHE_HUI_JIU_ZHU_1': 0,
    'cat__BEN_DI_FA_REN_0': 0,
    'cat__BEN_DI_FA_REN_1': 1,
    'cat__SI_WANG_ZHENG_MING_0': 1,
    'cat__SI_WANG_ZHENG_MING_1': 0,
    'cat__SAN_TIAN_SAN_JIAN_0': 0,
    'cat__SAN_TIAN_SAN_JIAN_1': 1,
    'cat__HU_JI_REN_KOU_ZC_0': 0,
    'cat__HU_JI_REN_KOU_ZC_1': 1,
    'cat__JIAO_NA_SHE_BAO_0': 0,
    'cat__JIAO_NA_SHE_BAO_1': 1,
    'cat__ZAI_XIAO_XUE_SHENG_0': 1,
    'cat__ZAI_XIAO_XUE_SHENG_1': 0,
    'cat__HU_JI_REN_KOU_ZX_0': 0,
    'cat__HU_JI_REN_KOU_ZX_1': 1,
    'cat__LIU_DONG_REN_KOU_ZX_0': 0,
    'cat__LIU_DONG_REN_KOU_ZX_1': 1,
    'cat__JU_ZHU_ZHENG_ZX_0': 0,
    'cat__JU_ZHU_ZHENG_ZX_1': 1,
    'num__SANSHI_TIAN_TING_CHE': 0,
    'num__BAN_NIAN_TING_CHE': 0,
    'num__SANSHI_TIAN_GONG_JIAO': 0,
    'num__BAN_NIAN_GONG_JIAO': 0,
}

# 系数
coefficients = {
    'cat__SAN_YUE_YI_YUAN_0': 0.0267841301056595,
    'cat__YI_NIAN_YI_YUAN_0': -0.06080960247184191,
    'cat__YI_NIAN_YI_YUAN_1': 0.08759373256686516,
    'cat__DANG_DI_JIAO_SHI_0': -0.03762272405289925,
    'cat__DANG_DI_JIAO_SHI_1': 0.06440685416470292,
    'cat__DANG_DI_XUE_SHENG_0': -2.377435151705934,
    'cat__DANG_DI_XUE_SHENG_1': 2.4042192819000783,
    'cat__DANG_DI_ZAI_ZHI_0': -4.426904400277336,
    'cat__DANG_DI_ZAI_ZHI_1': 4.453688530444914,
    'cat__GONG_JI_JIN_0': 0.0267841301056595,
    'cat__BU_DONG_CHAN_SHU_LIANG_0': 0.0267841301056595,
    'cat__GONG_ZU_FANG_0': -0.12491696066673845,
    'cat__GONG_ZU_FANG_1': 0.151701090769932,
    'cat__JIN_QI_JIAO_YI_0': 0.0267841301056595,
    'cat__SHE_HUI_JIU_ZHU_0': 0.29189152786988526,
    'cat__SHE_HUI_JIU_ZHU_1': -0.2651073977541228,
    'cat__BEN_DI_FA_REN_0': -0.1251567070372433,
    'cat__BEN_DI_FA_REN_1': 0.15194083714118123,
    'cat__SI_WANG_ZHENG_MING_0': 14.844202968478486,
    'cat__SI_WANG_ZHENG_MING_1': -14.817418838256215,
    'cat__SAN_TIAN_SAN_JIAN_0': -9.598264651956391,
    'cat__SAN_TIAN_SAN_JIAN_1': 9.625048782337968,
    'cat__HU_JI_REN_KOU_ZC_0': -7.6631054080441805,
    'cat__HU_JI_REN_KOU_ZC_1': 7.689889537958578,
    'cat__JIAO_NA_SHE_BAO_0': -4.426904400277336,
    'cat__JIAO_NA_SHE_BAO_1': 4.453688530444914,
    'cat__ZAI_XIAO_XUE_SHENG_0': -5.19084873912202,
    'cat__ZAI_XIAO_XUE_SHENG_1': 5.217632869220311,
    'cat__HU_JI_REN_KOU_ZX_0': 1.106552649943986,
    'cat__HU_JI_REN_KOU_ZX_1': -1.0797685198664433,
    'cat__LIU_DONG_REN_KOU_ZX_0': 1.421922288873072,
    'cat__LIU_DONG_REN_KOU_ZX_1': -1.3951381588103016,
    'cat__JU_ZHU_ZHENG_ZX_0': -0.08667567173478777,
    'cat__JU_ZHU_ZHENG_ZX_1': 0.1134598018397831,
    'num__SANSHI_TIAN_TING_CHE': 0.0714775774543422,
    'num__BAN_NIAN_TING_CHE': 0.0707182765968911,
    'num__SANSHI_TIAN_GONG_JIAO': -0.025570134132053852,
    'num__BAN_NIAN_GONG_JIAO': 0.0010907778535421545,
}

# 截距
intercept = 9.33747532

# 加权和
weighted_sum = sum(features[name] * coefficients[name] for name in features) + intercept

print("weighted_sum", weighted_sum)

# 逻辑函数
probability = 1 / (1 + np.exp(-weighted_sum))

print("Predicted probability:", probability)
