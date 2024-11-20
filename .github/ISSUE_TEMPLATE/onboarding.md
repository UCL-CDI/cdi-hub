---
name: "\U0001F951 Onboarding new users!" 
about: Suggested steps to onbard new users!
title: ''
labels: 'onboarding'
assignees: ''

---

## Onboarding new users
<!--Add a welcome message tagging github username -->
Welcome <ADD_YOUR_GITHUB_USER_NAME> to Central repository for CDI Reference Applications :tada:

As a new user, we recommend starting with the following items:  
* [ ] Add a title to the issue, usually: "Onboarding for Name"
* [ ] Create your ssh keys as suggested [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* [ ] Clone repo in your favorive path `git clone git@github.com:UCL-CDI/cdi-hub.git`
* [ ] If necessary, open a [new issue](https://github.com/UCL-CDI/cdi-hub/issues/new/choose) to submit a feature request or report a bug.  
* [ ] Create new branch: `git checkout -b ISSUENUMBER_FEATURE_BRANCH` (e.g., `git checkout -b 1-feature-branch`).  
* [ ] Commit and push a README.md with your own recipy:   
    ```
    git add .
    git commit -m 'short message #ISSUENUMBER' (e.g., `git commit -m 'my first commit #1'`)
    git push origin ISSUENUMBER-feature-branch (e.g. `git checkout -b 1-my-feature-branch`)
    ```
* [ ] Create a Pull Request, follow steps and ask for review to someone in our team.
