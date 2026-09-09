# StudentProjectAssignment-avt



	
### The Hungarian algorithm 
*Hungarian.py* takes the list of projects and their capacities in "Project List by Section.csv" and the students list of choices by "Project-Data.csv" and spits out "Project-assignment.csv"

We allow for projects with capacity > 1 by making project 87 into projects 87 & 87.1

### Penalties
If students have made invalid choices they have the cost penality incremented for each of their other choices for each invalid choice they made, this includes not making 10 choices. For instance if a student makes 9 choices instead of 10 then their first choice will be their second choice, and there will be no first choice. This doesnt mean they can't still get their first choice just then they will be penalized against other students who have the same first choice.

### Limitations:
- Assumes there is no project with capacity > 9

### Standalone app
*gui.py* is a Tkinter front end for the same `StudentProjectAssignment` logic — pick the two files, click "Run assignment", then "Save results as...". No typing paths, no console.

#### Input file templates
`templates/Projects-template.csv` and `templates/Student-choices-template.csv` show the exact columns each input file needs — copy one, replace the example rows with real data, and pass it to the app.

- **Projects file**: a Project ID column, then `Section` (the supervisor/theme) and `Capacity` (how many students that project can take — no need to list a project twice, capacity > 1 is split into slots automatically).
- **Student choices file**: a Student ID column, then one column per choice rank, most preferred first. Any number of choice columns works — a blank entry (or a project ID not in the projects file) counts as a missed choice and adds a small penalty to that student's *other* choices, per the Penalties section above.
- Column header wording is flexible (e.g. "Supervisor" or "Staff" both work for `Section`) — the app matches on keywords, not exact text. `.xlsx` files work too, including messier exports with the table located anywhere on the sheet.

#### Building the standalone app
Requires an environment with `pyinstaller` and this project's dependencies (`pip install -r requirements.txt`) — on this machine that's the `PDRA` micromamba env, which already has PyInstaller but needs `openpyxl` added (`pip install openpyxl`).

```
pyinstaller Hungarian.spec
```

This produces a single `dist/StudentProjectAssignment.exe` that runs without Python installed.

	

#### Results
~~~~
Result summary:
Number of students that got their first choice: 43
Number of students that got their second choice: 20
Number of students that got their third choice: 17
Average choice that each student got: 1.8470588235294119
~~~~

