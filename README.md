# sentiment-analysis
navigate to the working directory of this project 

for linux
```bash
cd path/to/Medical_app/
```
for windows
```bash
cd path\to\Medical_app\
```
### for linux os
### access for read&write
in your terminal / CMD run the following command to give docker permission to run bash script in the import.sh file
```bash
 chmod +x MedicalInfoDB/import.sh
```
to check that you have the permission run the following command 
```bash
ls -l MedicalInfoDB/import.sh
```
you suppose to see the following output

![Alt text](image-3.png)

### for windows os
### handle the conversion of line endings
in the Medical_App directory 
navigate to import.sh file that is exist in MedicalInfoDB folder

```bash
.\MedicalInfoDB\import.sh
```
1. **Open the Script File**: Open your script file (`import.sh`) in Visual Studio Code.

2. **Change Line Endings**:

    - Go to the bottom right corner of the window where it says the file encoding (e.g., UTF-8).
    ![Alt text](image-1.png)

    + Click on it and select "LF" (Unix) as the line ending format.

    ![alt text](image-4.png)

3. **Save the File**: After changing the line endings, save the file.


### running docker compose
now its time to run the compose file
```bash
docker-compose up -d
```

Now you can go to http://localhost/docs

You will see the automatic interactive API documentation (provided by Swagger UI):
![Alt text](image-2.png)

if you want to close the container run the following commadn

```bash
docker-compose down
```