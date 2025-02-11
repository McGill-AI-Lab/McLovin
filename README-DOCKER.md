
### USAGE : build the docker with `docker build -t mclovin .`

### Run with shell access : `docker run -it -p 8000:8000 mclovin`

### then run `python src/django_api/DatingApp/manage.py runserver 0.0.0.0:8000` for local

### or docker exec -it <container_id> python manage.py migrate

to get container id: docker ps
`
