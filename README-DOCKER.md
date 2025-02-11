
### USAGE : build the docker with `docker build -t mclovin .`
#### (`docker buildx build --platform linux/amd64 -t mclovin .` for my x64 server)


### Run with shell access : `docker run -it -p 8000:8000 mclovin`

### then run `python src/django_api/DatingApp/manage.py runserver 0.0.0.0:8000` for local

### or docker exec -it <container_id> python manage.py migrate

to get container id: docker ps

### NOTES for deployment to server

### export image `docker save -o mclovin.tar mclovin`

###

`
