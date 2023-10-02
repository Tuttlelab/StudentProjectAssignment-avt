from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm

import mimetypes
import os, pandas, sys, copy
import numpy as np
from scipy.optimize import linear_sum_assignment

def Hungarian(projects, choices):
    ChoiceMin = min(projects.index)
    ChoiceMax = max(projects.index)
    # We allow for projects with capacity > 1 by making project 87 into projects 87 & 87.1
    for i in projects.index:
        if projects.at[i, "Capacity"] > 1:
            for addition in range(1, projects.at[i, "Capacity"].astype(np.int64)):
                projects.at[i+(addition/10), "Capacity"] = 1    
                projects.at[i+(addition/10), "Section"] = projects.at[i, "Section"]    
            projects.at[i, "Capacity"] = 1

    projects = projects.sort_index()
    print(projects.iloc[-5:])

    choices.columns = [int(x) for x in choices.columns]
    Result = pandas.DataFrame(index=choices.index, columns=["Project"])
    print(choices)

    #Adjust the choices to account for projects with a capacity > 1
    DecimalChoices = copy.copy(choices)
    for student in choices.index:
        new_choice_list = []
        for choice in choices.loc[student].values:
            new_choice_list.append(choice)
            for decimal in np.arange(0.1, 1.0, 0.1):
                if float(choice)+decimal in projects.index:
                    new_choice_list.append(float(choice)+decimal)
                else:
                    break
        print(new_choice_list)
        for i in range(10):
            DecimalChoices.at[student, i+1] = new_choice_list[i]
    print(DecimalChoices)

    CostMatrix = pandas.DataFrame(index=DecimalChoices.index, columns=projects.index)
    #its a cost matrix so we want to price the choices low and the projects not chosen very highly
    HighCost = 10000
    CostMatrix[:] = HighCost

    for student in DecimalChoices.index:
        #Cost penality for invalid DecimalChoices
        CostPenalty = np.where((DecimalChoices.loc[student] < ChoiceMin) | (DecimalChoices.loc[student] > ChoiceMax))[0].shape[0]
        ValidChoices = np.where((DecimalChoices.loc[student] > ChoiceMin) & (DecimalChoices.loc[student] < ChoiceMax))[0]
        for cost,choice in enumerate(DecimalChoices.loc[student].iloc[ValidChoices]):
            CostMatrix.at[student, choice] = (cost**2) + CostPenalty
            #CostMatrix.at[student, choice] = cost + CostPenalty
    print(CostMatrix)

    row_ind, col_ind = linear_sum_assignment(CostMatrix)

    Result = pandas.DataFrame(index=choices.index, columns=["Project", "Choice #"])

    for i in range(len(row_ind)):
        student = CostMatrix.index[row_ind[i]]
        project = CostMatrix.columns[col_ind[i]]
        project = np.floor(project).astype(np.int64)
        Result.at[student, "Project"] = project
        try:
            Result.at[student, "Choice #"] = np.where(choices.loc[student] == project)[0][0]+1
        except:
            Result.at[student, "Choice #"] = -1
        
    print(Result)
    Result.to_csv("media/Project-assignment.csv")

    int2word = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fith",
                6: "Sixth", 7: "Seventh", 8:"Eigth", 9: "Nineth", 10: "Tenth"}
    Plotly = "<script>var data = [{"
    Plotly_x = []
    Plotly_y = []
    Report = "Result summary:<br>"
    for i in list(int2word.keys()):
        Report = Report + f"Number of students that got their {int2word[i]} choice: " + str((Result["Choice #"] == i).sum()) + "<br>"
        Plotly_x.append(int2word[i])
        Plotly_y.append((Result["Choice #"] == i).sum())
    Report = Report + "Average choice that each student got: " + str(Result[Result["Choice #"] != -1]["Choice #"].values.astype(np.float64).mean()) + "<br>"
    Report = Report + "Download results: <a href='media/Project-assignment.csv'>Project-assignment.csv</a>"
    Plotly = Plotly + 'x: ' + str(Plotly_x) + ","
    Plotly = Plotly + 'y: ' + str(Plotly_y) + ","
    Plotly = Plotly + "type: 'bar'}];"
    Plotly = Plotly + "Plotly.newPlot('myDiv', data); </script>"
    print(Plotly)
    return Report, Plotly

# def download_file(request):
#     # fill these variables with real values
#     fl_path = 'media/Project-assignment.csv'
#     filename = 'Project-assignment.csv'

#     fl = open(fl_path, 'r’)
#     mime_type, _ = mimetypes.guess_type(fl_path)
#     response = HttpResponse(fl, content_type=mime_type)
#     response['Content-Disposition'] = "attachment; filename=%s" % filename
#         return response

def home(request):
    os.system("python manage.py migrate")
    if request.method == 'POST':
        #print("request.FILES:", request.FILES)
        #print("request.FILES:", request.FILES.keys())
        if 'projectlists' in request.FILES.keys():
            myfile = request.FILES['projectlists']
            if os.path.exists("media/ProjectList.csv"):
                os.remove("media/ProjectList.csv")
            fs = FileSystemStorage()
            filename = fs.save("ProjectList.csv", myfile)
            #uploaded_file_url = fs.url(filename)
        elif 'studentchoices' in request.FILES.keys():
            myfile = request.FILES['studentchoices']
            if os.path.exists("media/StudentChoices.csv"):
                os.remove("media/StudentChoices.csv")
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
        projects = pandas.DataFrame()

    try:
        Choices = pandas.read_csv("media/StudentChoices.csv", index_col=0)
    except:
        Choices = pandas.DataFrame()

    if projects.shape[0] > 0 and Choices.shape[0] > 0:
        # We have the data to perform the Hungarian optimization
        try:
            report, plotly = Hungarian(projects, Choices)
        except:
            report = "Couldnt perform optimization on uploaded csv files"
            plotly = ""
    else:
        report = ""
        plotly = ""


    return render(request, 'core/home.html', { 'documents': documents, 
        'projects': projects.to_html(),
        'Choices': Choices.to_html(),
        'ProjectListFound': ProjectListFound,
        'StudentChoicesFound': StudentChoicesFound,
        'Report': report,
        'Plotly': plotly})



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
