#ifndef dm_analysis_h
#define dm_analysis_h

#include <TMath.h>
#include <algorithm>
#include <TLorentzVector.h>
#include <iostream>
#include "ROOT/RVec.hxx"
#include <Math/Vector3D.h>
#include <Math/VectorUtil.h>


using namespace ROOT;
using namespace ROOT::VecOps;

TLorentzVector boostinv(TLorentzVector q, TLorentzVector pboost) {
    TLorentzVector qprime(0.0, 0.0, 0.0, 0.0);

    double rmboost = pboost.E() * pboost.E() - pboost.X() * pboost.X() - pboost.Y() * pboost.Y() - pboost.Z() * pboost.Z();
    if (rmboost > 0.0) {
        rmboost = TMath::Sqrt(rmboost);
    }
    else {
        rmboost = 0.0;
    }

    double aux = (q.E() * pboost.E() - q.X() * pboost.X() - q.Y() * pboost.Y() - q.Z() * pboost.Z()) / rmboost;
    double aaux = (aux + q.E()) / (pboost.E() + rmboost);

    double qprimeE = aux;
    double qprimeX = q.X() - aaux * pboost.X();
    double qprimeY = q.Y() - aaux * pboost.Y();
    double qprimeZ = q.Z() - aaux * pboost.Z();

    qprime.SetPxPyPzE(qprimeX, qprimeY, qprimeZ, qprimeE);

    return qprime;
}

RVecF dphill_DH(
		RVecF Lepton_pt,
	        RVecF Lepton_eta,
		RVecF Lepton_phi,
	       	double PuppiMET_pt,
		double PuppiMET_phi) {
    TLorentzVector L1, L2, MET, Zp;

    L1.SetPtEtaPhiM(Lepton_pt[0], Lepton_eta[0], Lepton_phi[0], 0.);
    L2.SetPtEtaPhiM(Lepton_pt[1], Lepton_eta[1], Lepton_phi[1], 0.);
    MET.SetPtEtaPhiM(PuppiMET_pt, 0, PuppiMET_phi, 0.);
    

    ROOT::Math::PtEtaPhiEVector MET_vector;
    MET_vector.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    Zp = L1 + L2 + MET;

    ROOT::Math::XYZVector METvector;
    METvector = MET_vector.BoostToCM();
    
    ROOT::Math::XYZVector L1_vector;
    ROOT::Math::XYZVector L2_vector;

    ROOT::Math::PtEtaPhiEVector l1;
    ROOT::Math::PtEtaPhiEVector l2;
    ROOT::Math::PtEtaPhiEVector met;

    l1.SetCoordinates(L1.Pt(), L1.Eta(), L1.Phi(), L1.E());
    l2.SetCoordinates(L2.Pt(), L2.Eta(), L2.Phi(), L2.E());
    met.SetCoordinates(MET.Pt(), MET.Eta(), MET.Phi(), MET.E());

    L1_vector = ROOT::Math::VectorUtil::boost(l1, METvector);
    L2_vector = ROOT::Math::VectorUtil::boost(l2, METvector);

    float metpt = MET.Pt();
    float metphi_val = MET.Phi();
 
    float dPhillStar = ROOT::Math::VectorUtil::DeltaPhi(L1_vector, L2_vector); 


    ROOT::Math::PtEtaPhiEVector Zp_vector;
    Zp_vector = l1+l2+met;

    ROOT::Math::XYZVector Zpvector;
    Zpvector = Zp_vector.BoostToCM();
    
    ROOT::Math::XYZVector L1_vector_toZp;
    ROOT::Math::XYZVector L2_vector_toZp;
    ROOT::Math::XYZVector MET_vector_toZp;

    L1_vector_toZp = ROOT::Math::VectorUtil::boost(l1, Zpvector);
    L2_vector_toZp = ROOT::Math::VectorUtil::boost(l2, Zpvector);
    
    MET_vector_toZp = ROOT::Math::VectorUtil::boost(met, Zpvector);

    float dPhill_Zp = ROOT::Math::VectorUtil::Angle(L1_vector_toZp, L2_vector_toZp);

    float theta_1 = ROOT::Math::VectorUtil::Angle(L1_vector_toZp, Zp);
    float theta_2 = ROOT::Math::VectorUtil::Angle(L2_vector_toZp, Zp);

    float cos_theta1 = TMath::Cos(theta_1);
    float cos_theta2 = TMath::Cos(theta_2);

    float cos_theta = min(cos_theta1, cos_theta2);
 
    float theta_ll = ROOT::Math::VectorUtil::Angle(L1_vector_toZp+L2_vector_toZp, Zp);

    float dPhill_MET = ROOT::Math::VectorUtil::Angle(L1_vector_toZp+L2_vector_toZp, MET_vector);
    

    RVecF result = {dPhillStar, dPhill_Zp, cos_theta, theta_ll, dPhill_MET};

    return result;
}



#endif // dm_analysis_h




