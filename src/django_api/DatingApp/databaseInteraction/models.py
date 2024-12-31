from django.db import models

# this app was specifically made to test the database interactions of Django with the mongodb DB

class User(models.Model):
    
    # because the default __str__ method called when you print an instance of User is not user friendly (you would get <User: User object (1)> for ex), override it to print relevant information
    def __str__(self):
        return f"musician : id = {self.id}"
    
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    birth = models.DateField()
    
