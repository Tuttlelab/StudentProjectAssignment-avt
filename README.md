# StudentProjectAssignment-avt



	
### The Hungarian algorithm 
*Hungarian.py* takes the list of projects and their capacities in "Project List by Section.csv" and the students list of choices by "Project-Data.csv" and spits out "Project-assignment.csv"

We allow for projects with capacity > 1 by making project 87 into projects 87 & 87.1

### Penalties
If students have made invalid choices they have the cost penality incremented for each of their other choices for each invalid choice they made, this includes not making 10 choices. For instance if a student makes 9 choices instead of 10 then their first choice will be their second choice, and there will be no first choice. This doesnt mean they can't still get their first choice just then they will be penalized against other students who have the same first choice.

### Limitations:
- Assumes there is no project with capacity > 9

	

#### Results
~~~~
Result summary:
Number of students that got their first choice: 43
Number of students that got their second choice: 20
Number of students that got their third choice: 17
Average choice that each student got: 1.8470588235294119
~~~~

