from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm

import os, pandas, sys, copy
import numpy as np
from scipy.optimize import linear_sum_assignment

def Hungarian(projects, choices):
    ChoiceMin = min(projects.index)
    ChoiceMax = max(projects.index)
    # We allow for projects with capacity > 1 by making project 87 into projects 87 & 87.1
    for i in projects.index:
        if projects.at[i, "Capacity"] > 1:
            for addition in range(1, projects.at[i, "Capacity"].astype(np.int64) +1):
                projects.at[i+(addition/10), "Capacity"] = 1    
                projects.at[i+(addition/10), "Section"] = projects.at[i, "Section"]    
            projects.at[i, "Capacity"] = 1

    projects = projects.sort_index()
    print(projects.iloc[-5:])

    choices.columns = [int(x) for x in choices.columns]
    Result = pandas.DataFrame(index=choices.index, columns=["Project"])
    print(choices)


def home(request):
    os.system("python manage.py migrate")
    if request.method == 'POST':
        #print("request.FILES:", request.FILES)
        #print("request.FILES:", request.FILES.keys())
        if 'projectlists' in request.FILES.keys():
            myfile = request.FILES['projectlists']
            fs = FileSystemStorage()
            filename = fs.save("ProjectList.csv", myfile)
            #uploaded_file_url = fs.url(filename)
        elif 'studentchoices' in request.FILES.keys():
            myfile = request.FILES['studentchoices']
            fs = FileSystemStorage()
            filename = fs.save("StudentChoices.csv", myfile)


    #documents = Document.objects.all()
    documents = os.listdir("media")
    print("documents:", documents)

    if os.path.exists("media/ProjectList.csv"):
        ProjectListFound = "Project list found"
    else:
        ProjectListFound = "Project list NOT found"

    if os.path.exists("media/StudentChoices.csv"):
        StudentChoicesFound = "Student project choices found"
    else:
        StudentChoicesFound = "Student project choices NOT found"

    try:
        projects = pandas.read_csv("media/ProjectList.csv", index_col=0)
    except:
        projects = "None"

    try:
        Choices = pandas.read_csv("media/StudentChoices.csv", index_col=0)
    except:
        Choices = "None"

    if not isinstance(projects, str) and not isinstance(Choices, str):
        # We have the data to perform the Hungarian optimization
        Hungarian(projects, Choices)

    return render(request, 'core/home.html', { 'documents': documents, 
        'projects': projects.to_html(),
        'Choices': Choices.to_html(),
        'ProjectListFound': ProjectListFound,
        'StudentChoicesFound': StudentChoicesFound})



def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'core/simple_upload.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
