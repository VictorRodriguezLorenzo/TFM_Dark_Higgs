#ifndef EVALUATE_DNN_DARK_HIGGS
#define EVALUATE_DNN_DARK_HIGGS

#include <vector>
#include <iostream>
#include <TMath.h>
#include <math.h>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "ROOT/RVec.hxx"

#include <Python.h>


using namespace std;
using namespace ROOT;
using namespace ROOT::VecOps;

float evaluate_dnn(
    float lep_pt1,
    float lep_pt2,
    float lep_eta1,
    float lep_eta2,
    float mll,
    float mth,
    float mtw1,
    float mtw2,
    float ptll,
    float drll,
    float dphill,
    float PuppiMET_pt,
    float detall,
    float mpmet,
    float recoil,
    float mR,
    float mT2,
    float mTe,
    float mTi,
    float upara,
    float uperp,
    float dphilmet,
    float dphillmet,
    float mcoll,
    float mcollWW,
    float dPhillStar,
    float dPhill_Zp,
    float Cos_theta,
    float Theta_ll,
    float dPhill_MET,
    float first_btag_ID,
    float second_btag_ID
//    float ms,
//    float mZp,
//    float mx
)
{
    Py_Initialize();

    double result = -1;

    // Import the module
    PyObject* sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_DecodeFSDefault("/afs/cern.ch/user/v/victorr/private/DarkHiggs/Full2018_v7"));
    PyObject* pModule = PyImport_ImportModule("EvaluateDNN");

    if (pModule == NULL) {
        printf("ERROR importing module \n");
        exit(-1);
    } 

    if (pModule != NULL) {
        // Retrieve the function
        PyObject* pFunction = PyObject_GetAttrString(pModule, "load_neural_network");
        if (pFunction == NULL) {
            printf("ERROR getting function");
            exit(-1);
        }
	if (pFunction != NULL) {
            // Prepare arguments
	    std::vector<float> input;

            input.push_back(lep_pt1);
            input.push_back(lep_pt2);
            input.push_back(lep_eta1);
            input.push_back(lep_eta2);
            input.push_back(mll);
            input.push_back(mth);
            input.push_back(mtw1);
            input.push_back(mtw2);
            input.push_back(ptll);
            input.push_back(drll);
            input.push_back(dphill);
            input.push_back(PuppiMET_pt);
            input.push_back(detall);
            input.push_back(mpmet);
            input.push_back(recoil);
            input.push_back(mR);
            input.push_back(mT2);
            input.push_back(mTe);
            input.push_back(mTi);
            input.push_back(upara);
            input.push_back(uperp);
            input.push_back(dphilmet);
            input.push_back(dphillmet);
            input.push_back(mcoll);
            input.push_back(mcollWW);
            input.push_back(dPhillStar);
            input.push_back(dPhill_Zp);
            input.push_back(Cos_theta);
            input.push_back(Theta_ll);
            input.push_back(dPhill_MET);
            input.push_back(first_btag_ID);
            input.push_back(second_btag_ID);
//            input.push_back(ms);
//            input.push_back(mx);
//            input.push_back(mZp);

            // Input
            PyObject* pList = PyList_New(32);
            for (int i = 0; i < 32; ++i) {
                PyList_SetItem(pList, i, PyFloat_FromDouble((double)input[i]));
            }
            PyObject* pArgs = PyTuple_Pack(1, pList);
            if (pArgs != NULL) {
		    // Call the function
		    PyObject* pValue = PyObject_CallObject(pFunction, pArgs);
		    if (pValue != NULL) {
			    if (PyList_Check(pValue)) {
				    Py_ssize_t listSize = PyList_Size(pValue);
 
				    for (Py_ssize_t i = 0; i < listSize; i++) {
					    PyObject* listItem = PyList_GetItem(pValue, i);
					    result = PyFloat_AsDouble(listItem);
				    }
			    } else {
				    PyErr_Print();
			    } 
		    } else {
			    PyErr_Print();
		    }

		    Py_DECREF(pArgs);
	    } else {
		    PyErr_Print();
	    }

	    Py_DECREF(pFunction);
	} else {
		PyErr_Print();
	}

	Py_DECREF(pModule);
    } else {
	    PyErr_Print();
    }

//    cout << "Returning result DNN: " << result << endl;
    return (float)result;
    Py_Finalize();
}

#endif
