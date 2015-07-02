#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("final_project_dataset.pkl", "r"))
money, n_poi, n_sal, n_email, total_payments, n_poi_nan_pay = 0, 0, 0, 0, 0, 0
people = ("SKILLING JEFFREY K", "LAY KENNETH L","FASTOW ANDREW S") 
who = ""

for i in enron_data:
	if enron_data[i]["poi"]:
		n_poi += 1
		if enron_data[i]["total_payments"] == "NaN":
			n_poi_nan_pay += 1
	if enron_data[i]["email_address"] != "NaN":
		n_email += 1
	if enron_data[i]["salary"] != "NaN":
		n_sal += 1
	if enron_data[i]["total_payments"] == "NaN":
		total_payments += 1

for i in people:
	if money<enron_data[i]["total_payments"]:
		money = enron_data[i]["total_payments"]
		who = i

print "- How many data points (people) are in the dataset?\n+ %r" % len(enron_data)
print "- For each person, how many features are available?\n+ %r" % len(enron_data["SKILLING JEFFREY K"]) 
print "- How many POIs are there in the E+F dataset?\n+ %r" % n_poi
# check the poi_names.txt file
print "- How many POIs were there total?\n+ %r" % 35
print "- What is the total value of the stock belonging to James Prentice?\n+ %r" % enron_data["PRENTICE JAMES"]["total_stock_value"]
print "- How many email messages do we have from Wesley Colwell to persons of interest?\n+ %r" % enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "- What is the value of stock options exercised by Jeffrey Skilling?\n+ %r" % enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print "- Of these three individuals (Lay, Skilling and Fastow), who took home the most money?\n+ %r, %r"% (who, money)
print "- How many folks in this dataset have a quantified salary?\n+ %r" % n_sal
print "- What about a known email address?\n+ %r" % n_email
print "- How many people in the E+F dataset have NaN for their total payments?\n+ %r" % (total_payments / float(len(enron_data)))
print "- What percentage of POIs in the dataset hane 'NaN' for their payments?\n+ %r" % (n_poi_nan_pay / float(n_poi))
