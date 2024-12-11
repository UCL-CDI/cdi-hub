---
name: "\U0001F464 Onboarding new users!"
about: Suggested steps to onbard new users!
title: ''
labels: 'onboarding'
assignees: ''

---

## Onboarding new users
<!--Add a welcome message tagging github username -->
Welcome <ADD_YOUR_GITHUB_USER_NAME> to Central repository for CDI Reference Applications :tada:

We recommend covering the following points for your onboarding. 
Some may not be relevant, so feel free to skip them. If you have any questions, please add them to this issue or discuss them with someone on our team.

* [ ] Add a title to the issue, usually: "Onboarding for <ADD_YOUR_NAME>"
* [ ] Conduct a code and documentation walkthrough with a team member.
* [ ] Familiarise yourself with the GitHub workflow and [CONTRIBUTING](https://github.com/UCL-CDI/cdi-hub/blob/main/CONTRIBUTING.md) guidelines.
* [ ] Familiarise yourself with the [GLOSSARY](https://github.com/UCL-CDI/cdi-hub/blob/main/docs/glossary.md), [REFERENCES](https://github.com/UCL-CDI/cdi-hub/blob/main/docs/references.md), general [Infrastructure template](https://github.com/UCL-CDI/cdi-hub/tree/main/tutorials#infrastructure-overview) and consider updating it if you notice any missing terms or changes.
* [ ] Create your ssh keys as suggested [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* [ ] Clone repo in your favorive path `git clone git@github.com:UCL-CDI/cdi-hub.git`
* [ ] If necessary, open a [new issue](https://github.com/UCL-CDI/cdi-hub/issues/new/choose) to submit a feature request or report a bug.  
* [ ] Create new branch: `git checkout -b ISSUENUMBER_FEATURE_BRANCH` (e.g., `git checkout -b 1-feature-branch`).  
* [ ] [Local setup](https://github.com/UCL-CDI/cdi-hub/blob/main/tutorials/automatic-medical-image-reporting/docs/local-setup.md)
* [ ] [Setting up AWS](https://github.com/UCL-CDI/cdi-hub/blob/main/tutorials/automatic-medical-image-reporting/docs/aws-setup.md)
* [ ] Set up the `hello-ml` example to run both locally and on the AWS service. Follow the [GitHub workflow](https://github.com/UCL-CDI/cdi-hub/blob/main/CONTRIBUTING.md) to address any suggestions or feedback.
* [ ] Commit and push a README.md with your own recipy:   
    ```
    git add .
    git commit -m 'short message #ISSUENUMBER' (e.g., `git commit -m 'my first commit #1'`)
    git push origin ISSUENUMBER-feature-branch (e.g. `git checkout -b 1-my-feature-branch`)
    ```
* [ ] Create a Pull Request, follow steps and ask for review to someone in our team.
