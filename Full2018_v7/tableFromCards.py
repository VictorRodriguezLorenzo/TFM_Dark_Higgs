#!/usr/bin/env python3

import re
from sys import argv
import os.path
from optparse import OptionParser
from math import sqrt, fabs

parser = OptionParser()
parser.add_option("-s", "--stat", dest="stat", default=False, action="store_true")  # ignore systematic uncertainties to consider statistical uncertainties only
parser.add_option("-S", "--force-shape", dest="shape", default=False, action="store_true")  # ignore systematic uncertainties to consider statistical uncertainties only
parser.add_option("-a", "--asimov", dest="asimov", default=False, action="store_true")
parser.add_option("-m", "--mass", dest="mass", default=125, type="float")

(options, args) = parser.parse_args()
options.bin = True  # fake that is a binary output, so that we parse shape lines
options.noJMax = False
options.nuisancesToExclude = ''

from DatacardParser import *

DC = parseCard(file(args[0]), options)
nuisToConsider = [y for y in DC.systs]

errors = {}
for nuis in nuisToConsider:
    if nuis[2] == 'gmN':
        gmN = nuis[3][0]
    else:
        gmN = 0
    for channel in nuis[4]:
        if channel not in errors.keys():
            errors[channel] = {}
        for process in nuis[4][channel]:
            newError = 0.0
            if nuis[2] == 'gmN':
                gmN = nuis[3][0]
            if nuis[4][channel][process] == 0:
                continue
            if gmN != 0:
                newError = nuis[4][channel][process] * sqrt(gmN) / DC.exp[channel][process]
            else:
                if not isinstance(nuis[4][channel][process], float):
                    newError = fabs((nuis[4][channel][process][1] - nuis[4][channel][process][0]) / 2.0)  # symmetrized
                else:
                    newError = fabs(1 - nuis[4][channel][process])
            if process in errors[channel].keys():
                errors[channel][process] += newError * newError
            else:
                errors[channel][process] = newError * newError

for channel in errors:
    for process in errors[channel]:
        errors[channel][process] = sqrt(errors[channel][process])

for x in DC.exp:
    for y in DC.exp[x]:
        print(f"{x:>10} {y:>10} {DC.exp[x][y]:>10.2f} +/- {DC.exp[x][y] * errors[x][y]:>10.2f} ({errors[x][y] * 100:>10.2f} \\%%)")

size = "footnotesize"

print("\n")
print("=========================")
print("\n latex style \n")

print("\\begin{table}[h!]\\begin{center}")
print(f"\\{size}{{\\begin{{tabular}}", end=" ")

print("c|", end=" ")
for channel in DC.exp:
    print("c |", end=" ")
print("} \\hline")

print(" ", end=" ")
for channel in DC.exp:
    print(f"& {channel.replace('_', '-'):>13} ", end=" ")
print("\\\\ \\hline")

signals = DC.list_of_signals()
backgrounds = DC.list_of_backgrounds()

totsig = {}
errtotsig = {}
totbkg = {}
errtotbkg = {}

for s in signals:
    print(f" {s.replace('_', '-'):>13} ", end=" ")
    for channel in DC.exp:
        if s in DC.exp[channel].keys():  # possible that some backgrounds appear only in some channels
            print(f" & {DC.exp[channel][s]:>10.2f} +/- {DC.exp[channel][s] * errors[channel][s]:>10.2f} ({errors[channel][s] * 100:>10.0f} \\%%) ", end=" ")
            if channel not in totsig.keys():
                totsig[channel] = 0.0
            if channel not in errtotsig.keys():
                errtotsig[channel] = 0.0
            totsig[channel] = totsig[channel] + DC.exp[channel][s]
            errtotsig[channel] = errtotsig[channel] + (DC.exp[channel][s] * errors[channel][s] * DC.exp[channel][s] * errors[channel][s])
        else:
            print(" & - ", end=" ")
    print("\\\\")

print("\\hline")
print(f" {'signal':>13} ", end=" ")
for channel in DC.exp:
    errtotsig[channel] = sqrt(errtotsig[channel])
    print(f" & {totsig[channel]:>10.2f} +/- {errtotsig[channel]:>10.2f} ({errtotsig[channel] / totsig[channel] * 100:>10.0f} \\%%) ", end=" ")
print("\\\\")
print("\\hline")

for b in backgrounds:
    print(f" {b.replace('_', '-'):>13} ", end=" ")
    for channel in DC.exp:
        if b in DC.exp[channel].keys():  # possible that some backgrounds appear only in some channels
            print(f" & {DC.exp[channel][b]:>10.2f} +/- {DC.exp[channel][b] * errors[channel][b]:>10.2f} ({errors[channel][b] * 100:>10.0f} \\%%) ", end=" ")
            if channel not in totbkg.keys():
                totbkg[channel] = 0.0
            if channel not in errtotbkg.keys():
                errtotbkg[channel] = 0.0
            totbkg[channel] = totbkg[channel] + DC.exp[channel][b]
            errtotbkg[channel] = errtotbkg[channel] + (DC.exp[channel][b] * errors[channel][b] * DC.exp[channel][b] * errors[channel][b])
        else:
            print(" & - ", end=" ")
    print("\\\\")

print("\\hline")
print(f" {'background':>13} ", end=" ")
for channel in DC.exp:
    errtotbkg[channel] = sqrt(errtotbkg[channel])
    print(f" & {totbkg[channel]:>10.2f} +/- {errtotbkg[channel]:>10.2f} ({errtotbkg[channel] / totbkg[channel] * 100:>10.0f}\\%%) ", end=" ")
print("\\\\")
print("\\hline")

print("\\end{tabular}")
print("}")
print("\\end{center}")

print("\n\n\n")

print("\\end{table}")

print("=========================")
print("\n\n\n")
