mkdir resources
Invoke-WebRequest `
    -Uri https://github.com/tonsky/FiraCode/releases/download/6.2/Fira_Code_v6.2.zip `
    -OutFile resources/firacode.zip

cd resources
Expand-Archive firacode.zip fira_code
rm firacode.zip
cd ..
