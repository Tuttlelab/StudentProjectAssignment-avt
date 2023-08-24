from django.http import HttpResponse
import os
from django.shortcuts import render
from .forms import CsvUploadForm

def index(request):
    if not os.path.exists("csv_cache"):
        os.mkdir("csv_cache")
    return HttpResponse("Hello, world. You're at the polls index.")

# def upload_csv(request):
#     if request.method == 'POST':
#         form = CsvUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             # Handle successful upload
#     else:
#         form = CsvUploadForm()
#     return render(request, 'upload_csv.html', {'form': form})