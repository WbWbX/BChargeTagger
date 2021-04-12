import tensorflow as tf

featureDict = {
    "global": {
        "branches": [
            "global_pt",
            "global_eta",
            "global_phi",
            "global_mass",
            "global_energy",

            "global_area",

            "global_beta",
            "global_dR2Mean",
            "global_frac01",
            "global_frac02",
            "global_frac03",
            "global_frac04",

            "global_jetR",
            "global_jetRchg",

            "global_n60",
            "global_n90",

            "global_chargedEmEnergyFraction",
            "global_chargedHadronEnergyFraction",
            "global_chargedMuEnergyFraction",
            "global_electronEnergyFraction",

            "global_tau1",
            "global_tau2",
            "global_tau3",

            "global_relMassDropMassAK",
            "global_relMassDropMassCA",
            "global_relSoftDropMassAK",
            "global_relSoftDropMassCA",

            "global_thrust",
            "global_sphericity",
            "global_circularity",
            "global_isotropy",
            "global_eventShapeC",
            "global_eventShapeD",

            "global_numberCpf",
            "global_numberNpf",
            "global_numberSv",
            "global_numberMuon",
            "global_numberElectron",

            "csv_trackSumJetEtRatio",
            "csv_trackSumJetDeltaR",
            "csv_vertexCategory",
            "csv_trackSip2dValAboveCharm",
            "csv_trackSip2dSigAboveCharm",
            "csv_trackSip3dValAboveCharm",
            "csv_trackSip3dSigAboveCharm",
            "csv_jetNTracksEtaRel",
            "csv_jetNSelectedTracks",
        ],
        "preprocessing":{
            "global_pt":lambda x: tf.math.log(tf.clip_by_value(x,1e-3,100.)),
            "global_mass":lambda x: tf.math.log(tf.nn.relu(x)+1e-3),
            "csv_trackSip2dValAboveCharm":lambda x: tf.math.sign(x)*(tf.math.log(tf.math.abs(x)+1e-3)+5),
            "csv_trackSip2dSigAboveCharm":lambda x: tf.math.log(tf.math.abs(x)+1e-3),
            "csv_trackSip3dValAboveCharm":lambda x: tf.math.sign(x)*(tf.math.log(tf.math.abs(x)+1e-3)+5),
            "csv_trackSip3dSigAboveCharm":lambda x: tf.math.log(tf.math.abs(x)+1e-3),
        },
    },
    
    "cpf_charge": {
        "branches": [
            "cpf_charge",
        ],
        "length":"length_cpf",
        "offset":"length_cpf_offset",
        "max": 25
    },
    
    "cpf": {
        "branches": [

            "cpf_ptrel",
            "cpf_deta",
            "cpf_dphi",
            "cpf_deltaR",
            
            "cpf_trackEtaRel",
            "cpf_trackPtRel",
            "cpf_trackPPar",
            "cpf_trackDeltaR",
            "cpf_trackPParRatio",
            "cpf_trackPtRatio",
            "cpf_trackSip2dVal",
            "cpf_trackSip2dSig",
            "cpf_trackSip3dVal",
            "cpf_trackSip3dSig",
            "cpf_trackJetDistVal",
            "cpf_trackJetDistSig",
            "cpf_drminsv",
            "cpf_vertex_association",
            "cpf_fromPV",
            "cpf_puppi_weight",
            "cpf_track_chi2",
            "cpf_track_quality",
            "cpf_track_numberOfValidPixelHits",
            "cpf_track_pixelLayersWithMeasurement",
            "cpf_track_numberOfValidStripHits",
            "cpf_track_stripLayersWithMeasurement",
            "cpf_relmassdrop",

            "cpf_trackSip2dValSV",
            "cpf_trackSip2dSigSV",
            "cpf_trackSip3dValSV",
            "cpf_trackSip3dSigSV",

            "cpf_matchedMuon",
            "cpf_matchedElectron",
            "cpf_matchedSV",
            "cpf_track_ndof",

            "cpf_dZmin"
        ],
        "preprocessing":{
            "cpf_ptrel":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
            "cpf_deta":lambda x: tf.math.abs(x),
            "cpf_dphi":lambda x: tf.math.abs(x),

            "cpf_trackEtaRel":lambda x: tf.math.log(1+tf.math.abs(x)),
            "cpf_trackPtRel":lambda x: tf.math.log(1e-1+tf.nn.relu(1-x)),
            "cpf_trackPPar": lambda x: tf.math.log(1e-2+tf.nn.relu(x)),
            "cpf_trackPParRatio": lambda x: tf.math.log(1e-4+tf.nn.relu(1-x))*0.1,
            "cpf_track_chi2":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "cpf_trackDeltaR":lambda x: 0.1/(0.1+tf.nn.relu(x)),
            "cpf_trackJetDistVal": lambda x: tf.math.log(tf.nn.relu(-x)+1e-5),
            "cpf_trackSip2dVal":lambda x: tf.math.sign(x)*(tf.math.log(tf.math.abs(x)+1e-3)+5),
            "cpf_trackSip2dSig":lambda x: tf.math.log(tf.math.abs(x)+1e-3),
            "cpf_trackSip3dVal":lambda x: tf.math.sign(x)*(tf.math.log(tf.math.abs(x)+1e-3)+5),
            "cpf_trackSip3dSig":lambda x: tf.math.log(tf.math.abs(x)+1e-3),
            "cpf_track_ndof":lambda x: x*0.05,

            "cpf_dZmin": lambda x: tf.math.log(tf.math.abs(x)+1e-6)
        },
        "length":"length_cpf",
        "offset":"length_cpf_offset",
        "max": 25
    },

    "npf": {
        "branches": [
            "npf_ptrel",
            "npf_deta",
            "npf_dphi",
            "npf_deltaR",
            "npf_isGamma",
            "npf_hcal_fraction",
            "npf_drminsv",
            "npf_puppi_weight",
            "npf_relmassdrop",
        ],
        "preprocessing":{
            "npf_ptrel":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
            "npf_deta":lambda x: tf.math.abs(x),
            "npf_dphi":lambda x: tf.math.abs(x),

            "npf_deltaR":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),

        },
        "length":"length_npf",
        "offset":"length_npf_offset",
        "max": 25
    },

    "sv": {
        "branches": [
            "sv_ptrel",
            "sv_deta",
            "sv_dphi",
            "sv_deltaR",
            "sv_mass",
            "sv_ntracks",
            "sv_chi2",
            "sv_ndof",
            "sv_dxy",
            "sv_dxysig",
            "sv_d3d",
            "sv_d3dsig",
            "sv_costhetasvpv",
            "sv_enratio"
        ],
        "preprocessing":{
            "sv_ptrel":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
            "sv_deta":lambda x: tf.math.abs(x),
            "sv_dphi":lambda x: tf.math.abs(x),
            "sv_chi2":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "sv_deltaR":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),

            "sv_dxy":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
            "sv_dxysig":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
            "sv_d3d":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
            "sv_d3dsig":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
        },
        "length":"length_sv",
        "offset":"length_sv_offset",
        "max": 4
    },
    
    "muon_charge": {
        "branches": [
            "muon_charge",
        ],
        "length":"length_mu",
        "offset":"length_mu_offset",
        "max": 2
    },
    
    "muon": {
        "branches": [
            "muon_ptrel",
            "muon_deta",
            "muon_dphi",
            
            "muon_energy",
            "muon_et",
            "muon_deltaR",
            "muon_numberOfMatchedStations",

            "muon_IP2d",
            "muon_IP2dSig",
            "muon_IP3d",
            "muon_IP3dSig",

            "muon_EtaRel",
            "muon_dxy",
            "muon_dxyError",
            "muon_dxySig",
            "muon_dz",
            "muon_dzError",
            "muon_dzSig",
            "muon_numberOfValidPixelHits",
            "muon_numberOfpixelLayersWithMeasurement",
            "muon_numberOfstripLayersWithMeasurement",

            "muon_chi2",
            "muon_ndof",

            "muon_caloIso",
            "muon_ecalIso",
            "muon_hcalIso",

            "muon_sumPfChHadronPt",
            "muon_sumPfNeuHadronEt",
            "muon_Pfpileup",
            "muon_sumPfPhotonEt",

            "muon_sumPfChHadronPt03",
            "muon_sumPfNeuHadronEt03",
            "muon_Pfpileup03",
            "muon_sumPfPhotonEt03",


            "muon_timeAtIpInOut",
            "muon_timeAtIpInOutErr",
            "muon_timeAtIpOutIn"
        ],
        "preprocessing":{
            "muon_ptrel":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
            "muon_deta": lambda x: tf.math.abs(x),
            "muon_dphi": lambda x: tf.math.abs(x),
            "muon_deltaR":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),

            "muon_IP2d":lambda x: tf.math.sign(x)*(tf.math.log(tf.math.abs(x)+1e-3)+5),
            "muon_IP2dSig":lambda x: tf.math.log(tf.math.abs(x)+1e-3),
            "muon_IP3d":lambda x: tf.math.sign(x)*(tf.math.log(tf.math.abs(x)+1e-3)+5),
            "muon_IP3dSig":lambda x: tf.math.log(tf.math.abs(x)+1e-3),

            "muon_dxy":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "muon_dxySig":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "muon_dz":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "muon_dzError":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "muon_dzSig":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),

            "muon_chi2":lambda x: tf.math.log(tf.minimum(1e3,tf.nn.relu(x))+1e-6),
            "muon_ndof": lambda x: 0.1*x,

            "muon_caloIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_ecalIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_hcalIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_sumPfChHadronPt":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_sumPfNeuHadronEt":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_Pfpileup":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_sumPfPhotonEt":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),

            "muon_sumPfChHadronPt03":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_sumPfNeuHadronEt03":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_Pfpileup03":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "muon_sumPfPhotonEt03":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),

            "muon_timeAtIpInOut":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "muon_timeAtIpInOutErr":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "muon_timeAtIpOutIn":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x))
        },
        "length":"length_mu",
        "offset":"length_mu_offset",
        "max": 2
    },
    
    'electron_charge': {
        "branches": [
            "electron_charge",
        ],
        "length":"length_ele",
        "offset":"length_ele_offset",
        "max": 2
    },

    "electron": {
        "branches": [
            "electron_ptrel",
            "electron_deltaR",
            "electron_deta",
            "electron_dphi",
            "electron_energy",
            "electron_EtFromCaloEn",
            "electron_isEB",
            "electron_isEE",
            "electron_ecalEnergy",
            "electron_isPassConversionVeto",
            "electron_convDist",
            "electron_convFlags",
            "electron_convRadius",
            "electron_hadronicOverEm",
            "electron_ecalDrivenSeed",
            "electron_IP2d",
            "electron_IP2dSig",
            "electron_IP3d",
            "electron_IP3dSig",

            "electron_elecSC_energy",
            "electron_elecSC_deta",
            "electron_elecSC_dphi",
            "electron_elecSC_et",
            "electron_elecSC_eSuperClusterOverP",
            #"electron_scPixCharge",
            "electron_superClusterFbrem",

            "electron_eSeedClusterOverP",
            "electron_eSeedClusterOverPout",
            "electron_eSuperClusterOverP",

            "electron_sigmaEtaEta",
            "electron_sigmaIetaIeta",
            "electron_sigmaIphiIphi",
            "electron_e5x5",
            "electron_e5x5Rel",
            "electron_e1x5Overe5x5",
            "electron_e2x5MaxOvere5x5",
            "electron_r9",
            "electron_hcalOverEcal",
            "electron_hcalDepth1OverEcal",
            "electron_hcalDepth2OverEcal",

            "electron_deltaEtaEleClusterTrackAtCalo",
            "electron_deltaEtaSeedClusterTrackAtCalo",
            "electron_deltaPhiSeedClusterTrackAtCalo", 
            "electron_deltaEtaSeedClusterTrackAtVtx",
            "electron_deltaEtaSuperClusterTrackAtVtx",
            "electron_deltaPhiEleClusterTrackAtCalo",
            "electron_deltaPhiSuperClusterTrackAtVtx",

            "electron_sCseedEta",

            "electron_EtaRel",
            "electron_dxy",
            "electron_dxyError",
            "electron_dxySig",
            "electron_dz",
            "electron_dzError",
            "electron_dzSig",
            "electron_nbOfMissingHits",
            #"electron_gsfCharge",
            "electron_ndof",
            "electron_chi2",
            "electron_numberOfBrems",
            "electron_fbrem",

            "electron_neutralHadronIso",
            "electron_particleIso",
            "electron_photonIso",
            "electron_puChargedHadronIso",
            "electron_trackIso",
            "electron_ecalPFClusterIso",
            "electron_hcalPFClusterIso",

            "electron_pfSumPhotonEt",
            "electron_pfSumChargedHadronPt", 
            "electron_pfSumNeutralHadronEt",
            "electron_pfSumPUPt",

            "electron_dr04TkSumPt",
            "electron_dr04EcalRecHitSumEt",
            "electron_dr04HcalDepth1TowerSumEt",
            "electron_dr04HcalDepth1TowerSumEtBc",
            "electron_dr04HcalDepth2TowerSumEt",
            "electron_dr04HcalDepth2TowerSumEtBc",
            "electron_dr04HcalTowerSumEt",
            "electron_dr04HcalTowerSumEtBc"
        ],
        "preprocessing":{
            "electron_ptrel":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),
            "electron_deta": lambda x: tf.math.abs(x),
            "electron_dphi": lambda x: tf.math.abs(x),
            "electron_deltaR":lambda x: tf.math.log(1e-6+tf.nn.relu(x)),

            "electron_chi2":lambda x: tf.math.log(tf.minimum(1e3,tf.nn.relu(x))+1e-6),
            "electron_ndof": lambda x: x*0.1,

            "electron_hadronicOverEm": lambda x: tf.math.log(1./(1e-2+tf.nn.relu(x))),
            "electron_eSeedClusterOverP":lambda x: tf.math.log(1e-5+tf.nn.relu(1-x)),
            "electron_eSeedClusterOverPout":lambda x: tf.math.log(1e-5+tf.nn.relu(1-x)),

            "electron_elecSC_eSuperClusterOverP":lambda x: tf.math.log(1e-5+tf.nn.relu(x)),
            "electron_eSuperClusterOverP":lambda x: tf.math.log(1e-5+tf.nn.relu(x)),

            "electron_IP2d":lambda x: tf.math.sign(x)*(tf.math.log(tf.math.abs(x)+1e-3)+5),
            "electron_IP2dSig":lambda x: tf.math.log(tf.math.abs(x)+1e-3),
            "electron_IP3d":lambda x: tf.math.sign(x)*(tf.math.log(tf.math.abs(x)+1e-3)+5),
            "electron_IP3dSig":lambda x: tf.math.log(tf.math.abs(x)+1e-3),

            "electron_dxy":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "electron_dxyError":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "electron_dxySig":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "electron_dz":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "electron_dzError":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),
            "electron_dzSig":lambda x: tf.math.sign(x)*tf.math.log(1e-6+tf.math.abs(x)),

            "electron_neutralHadronIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "electron_photonIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "electron_puChargedHadronIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "electron_trackIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "electron_ecalPFClusterIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "electron_hcalPFClusterIso":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),

            "electron_hcalDepth1OverEcal":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
            "electron_hcalDepth2OverEcal":lambda x: tf.math.log(tf.nn.relu(x)+1e-6),
        },
        "length":"length_ele",
        "offset":"length_ele_offset",
        "max": 2,
        
    }
}
