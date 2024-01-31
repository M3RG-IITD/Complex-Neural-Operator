from models import LSM_2D, LSM_3D, LSM_Irregular_Geo, FNO_2D, FNO_3D, FNO_Irregular_Geo, FFNO_2D, FFNO_2D_Irregular_Geo, complex_FFNO_2D, complex_FFNO_2D_Irregular_Geo, complex_FFNO_3D, con_cono, complete_complex

def get_model(args):
    model_dict = {
        'FNO_2D': FNO_2D,
        'FNO_3D': FNO_3D,
        'FNO_Irregular_Geo': FNO_Irregular_Geo,
        'LSM_2D': LSM_2D,
        'LSM_3D': LSM_3D,
        'LSM_Irregular_Geo': LSM_Irregular_Geo,
        'FFNO_2D': FFNO_2D,
        'FFNO_2D_Irregular_Geo': FFNO_2D_Irregular_Geo,
        'complex_FFNO_2D': complex_FFNO_2D,
        'complex_FFNO_2D_Irregular_Geo': complex_FFNO_2D_Irregular_Geo,
        'complex_FFNO_3D': complex_FFNO_3D,
        'con_cono': con_cono,
        'complete_complex': complete_complex
    }
    
    if args.model == 'LSM_Irregular_Geo' or args.model == 'FNO_Irregular_Geo' or args.model == 'FFNO_2D_Irregular_Geo' or args.model == 'complex_FFNO_2D_Irregular_Geo':
        return model_dict[args.model].Model(args).cuda(), model_dict[args.model].IPHI().cuda()
    else:
        return model_dict[args.model].Model(args).cuda()
